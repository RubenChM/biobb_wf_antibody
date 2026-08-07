"""Plotting helpers used by the biobb_antibody notebook"""

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly import subplots


def load_xvg(xvg_path):
    """Load every column of a GROMACS XVG file, skipping the grace comment lines"""
    return np.loadtxt(str(xvg_path), comments=['#', '@'])

def read_xvg(xvg_path):
    """Read the first data column of a GROMACS XVG file"""
    data = load_xvg(xvg_path)
    return data[:, 0], data[:, 1]

def plot_xvg(xvg_paths, title, ytitle, xtitle="Time (ps)", per_frame=False, mode='lines',
             yscale=1.0, ymax=None, height=None):
    """Plot one or several XVG files ({legend: path}) in a single figure

    Set per_frame=True for concatenated trajectories, where the time column
    restarts on every block and is therefore not a continuous simulation time.
    Use mode='markers' for groups that are not contiguous, so that no line is
    drawn across the atoms that do not belong to the group.
    GROMACS reports distances in nm, use yscale=10 to plot them in angstroms.
    Set ymax to drop the points above it, useful to skip the huge energies of
    the first minimization steps.
    """
    if per_frame:
        xtitle = "Frame"

    traces = []
    for name, xvg_path in xvg_paths.items():
        x, y = read_xvg(xvg_path)
        if ymax is not None:
            mask = y < ymax
            x, y = x[mask], y[mask]
        if per_frame:
            x = np.arange(len(y))
        traces.append(go.Scatter(x=x, y=y * yscale, name=name, mode=mode))

    plotly.offline.init_notebook_mode(connected=True)

    fig = {
        "data": traces,
        "layout": go.Layout(title=title,
                            xaxis=dict(title=xtitle),
                            yaxis=dict(title=ytitle),
                            height=height
                           )
    }

    plotly.offline.iplot(fig)

def plot_xvg_columns(xvg_path, title, ytitles, xtitle="Time (ps)"):
    """Plot each data column of a single XVG file in its own side by side panel

    ytitles labels the columns in order, so its length sets the number of panels.
    """
    data = load_xvg(xvg_path)

    fig = subplots.make_subplots(rows=1, cols=len(ytitles), print_grid=False)
    for col, ytitle in enumerate(ytitles, start=1):
        fig.append_trace(go.Scatter(x=data[:, 0], y=data[:, col]), 1, col)
        fig['layout'][f'xaxis{col}'].update(title=xtitle)
        fig['layout'][f'yaxis{col}'].update(title=ytitle)
    fig['layout'].update(title=title, showlegend=False)

    plotly.offline.init_notebook_mode(connected=True)
    plotly.offline.iplot(fig)

def _smooth_density(values, grid, bandwidth=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.zeros_like(grid)

    if bandwidth is None:
        span = np.ptp(values)
        bandwidth = max(np.std(values) / 3, 0.05 * (span if span > 0 else 1.0))
        bandwidth = max(bandwidth, 1e-3)

    x = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * x**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    max_density = density.max()
    if max_density > 0:
        density = density / max_density
    return density


def plot_dockq_vs_score(dir, ax = None, colors = ['blue', 'orange'], label = None):
    results_em = dir / 'output/run/08_caprieval/capri_ss.tsv'
    results_clust = dir / 'output/run/11_caprieval/capri_ss.tsv'
    df_em = pd.read_csv(results_em, sep='\t')
    df_clust = pd.read_csv(results_clust, sep='\t')

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.set_title('DockQ vs HADDOCK score')
    if fig._suptitle is not None:
        fig._suptitle.set_text('')

    if not getattr(ax, '_dockq_hist_ready', False):
        ax_histx = ax.inset_axes([0.0, 1.02, 1.0, 0.18], transform=ax.transAxes)
        ax_histy = ax.inset_axes([1.02, 0.0, 0.18, 1.0], transform=ax.transAxes)
        ax_histx.set_facecolor('none')
        ax_histy.set_facecolor('none')
        ax_histx.tick_params(labelbottom=False)
        ax_histy.tick_params(labelleft=False)
        ax._dockq_hist_ready = True
        ax._dockq_histx = ax_histx
        ax._dockq_histy = ax_histy

    ax.scatter(df_em['score'], df_em['dockq'], color=colors[0], alpha=0.7, s=40, label='_nolegend_')
    ax.scatter(df_clust['score'], df_clust['dockq'], color=colors[1], alpha=0.7, s=40, label='_nolegend_', marker='x')
    ax.set_xlabel('HADDOCK score')
    ax.set_ylabel('DockQ')

    for threshold in [0.23, 0.49, 0.80]:
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(ax.get_xlim()[1], threshold, f' {threshold}', va='bottom', ha='right', color='gray', fontsize=8)

    # Fit linear regression lines for EM and Cluster data
    for label_line, df in [('EM', df_em)]:#, ('Cluster', df_clust)]:
        x = df['score'].astype(float)
        y = df['dockq'].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            poly = np.poly1d(coeffs)
            xs = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xs, poly(xs), color=colors[0] if label_line == 'EM' else colors[1], linewidth=2, alpha=0.8, linestyle='--')

    def _plot_density(ax_panel, values, color, orientation='x'):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            return

        grid = np.linspace(values.min(), values.max(), 200)
        density = _smooth_density(values, grid)

        if orientation == 'x':
            ax_panel.plot(grid, density, color=color, linewidth=2.5, alpha=1.0, zorder=3)
            ax_panel.fill_between(grid, density, 0, color=color, alpha=0.20, zorder=2)
        else:
            ax_panel.plot(density, grid, color=color, linewidth=2.5, alpha=1.0, zorder=3)
            ax_panel.fill_betweenx(grid, density, 0, color=color, alpha=0.20, zorder=2)

    # ax._dockq_histx.cla()
    # ax._dockq_histy.cla()
    _plot_density(ax._dockq_histx, df_em['score'], colors[0], orientation='x')
    # _plot_density(ax._dockq_histx, df_clust['score'], colors[1], orientation='x')
    ax._dockq_histx.set_ylabel('Density')
    ax._dockq_histx.set_xlabel('')
    ax._dockq_histx.set_xlim(ax.get_xlim())
    ax._dockq_histx.set_yticks([])

    _plot_density(ax._dockq_histy, df_em['dockq'], colors[0], orientation='y')
    # _plot_density(ax._dockq_histy, df_clust['dockq'], colors[1], orientation='y')
    ax._dockq_histy.set_xlabel('Density')
    ax._dockq_histy.set_ylabel('')
    ax._dockq_histy.set_ylim(ax.get_ylim())
    ax._dockq_histy.set_xticks([])

    if label is not None:
        ax.plot([], [], color=colors[0], linewidth=2, label=label)
        ax.legend(loc='best')

    return ax
