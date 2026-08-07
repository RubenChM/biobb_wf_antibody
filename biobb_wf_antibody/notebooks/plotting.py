"""Plotting and structure-visualization helpers used by the biobb_antibody notebook"""

import ipywidgets
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import MDAnalysis as mda
import nglview as nv
from IPython.display import display, Markdown
from MDAnalysis.analysis import align
from plotly import subplots

from cdr import CDR_RANGES


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


def show_clusters(cluster_pdb, aligned_pdb, anarcii_pdb, loop_ranges=CDR_RANGES):
    """Align the cluster representatives of `cluster_pdb` and show them in one NGL view.

    `gmx cluster` writes a single multi-model PDB, one model per cluster, and the models are not
    superposed on each other. They are aligned here, written to `aligned_pdb`, and displayed
    overlaid with a dropdown to isolate each cluster. Returns the NGL widget.
    """

    def on_dropdown_change(change):
        """Show all clusters or isolate the selected one."""
        if change['type'] == 'change' and change['name'] == 'value':
            selected = change['new']
            print(f"Selected cluster: {selected}")
            target = "*" if selected == 'All' else f"/{int(selected.split()[-1]) - 1}"
            view._remote_call('setSelection', target='compList',
                              args=[target], kwargs=dict(component_index=0))

    # Locate the loops in the representatives. anarcii_pdb carries the IMGT numbering
    # (non-continuous, gaps at unused positions) while the representatives were renumbered
    # continuously by pdb2gmx; residue order is preserved between the two, so the loops are mapped
    # by resindex, not by resid.
    u_anarcii = mda.Universe(str(anarcii_pdb))
    loop_resindices = u_anarcii.select_atoms(
        " or ".join(f"resid {s}-{e}" for s, e in loop_ranges)).residues.resindices
    loop_resindex_sel = "resindex " + " ".join(map(str, loop_resindices))

    # Superpose every representative on the first one using the backbone *outside* the loops, i.e.
    # the framework. The clustering ran with 'nofit' on a framework-fitted trajectory, so the
    # representatives already share that frame and this only takes out the residual offset; fitting
    # on the loops instead would superpose away the very differences the view is meant to show.
    mobile = mda.Universe(str(cluster_pdb))
    ref_u = mda.Universe(str(cluster_pdb))
    ref_u.trajectory[0]
    align.AlignTraj(mobile, ref_u, select=f"backbone and not ({loop_resindex_sel})",
                    filename=str(aligned_pdb)).run()
    n_models = len(mobile.trajectory)

    # Highlight the loops by absolute atom index ('@' in the NGL selection language). Neither the
    # chain nor the residue number can be used: the representatives come from the simulation
    # topology, so every atom lands in a single chain, and pdb2gmx numbering restarts on the second
    # chain, leaving duplicated residue numbers.
    #
    # NGL keeps every model of the file in one structure and numbers their atoms consecutively, so
    # the indices of the first model have to be repeated for each of the following ones, shifted by
    # a whole model. Without the shift only the first cluster is highlighted.
    u_display = mda.Universe(str(aligned_pdb))
    loop_ix, n_atoms = u_display.select_atoms(loop_resindex_sel).ix, u_display.atoms.n_atoms
    cdr_selection = "@" + ",".join(str(ix + model * n_atoms)
                                   for model in range(n_models) for ix in loop_ix)

    # Dropdown to pick a single cluster (or all of them)
    opts = ['All'] + [f"Cluster {i + 1}" for i in range(n_models)]
    mdsel = ipywidgets.Dropdown(options=opts, description='Cluster:', disabled=False)

    # Overlay every cluster: full structure in grey, CDR loops highlighted
    view = nv.show_structure_file(str(aligned_pdb))
    view.add_cartoon(color='lightgrey', opacity=0.4)
    view.add_licorice(selection=cdr_selection, color="green")
    view._remote_call('setSize', target='Widget', args=['', '600px'])
    view.layout.margin = "auto"
    view.camera = 'orthographic'
    view._remote_call('setSelection', target='compList', args=['*'], kwargs=dict(component_index=0))

    mdsel.observe(on_dropdown_change, names='value')
    # The cutoff, not a target count, decides how many clusters come out
    display(Markdown(f"##### {n_models} CDR-loop cluster{'s' if n_models > 1 else ''}, "
                     "loops in green. Select a cluster:"))
    display(mdsel)
    return view
