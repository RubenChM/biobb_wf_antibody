# pixi-pack instructions
``` shell
cd ../conda_env
pixi lock
./build_haddock3_wheel.sh           # see "haddock3" below; only when its version changes
pixi-pack --create-executable --inject haddock3-*.whl
scp environment.sh user@<cluster>:/path/to/biobb_wf_antibody/
# On cluster, run:
./environment.sh                    # unpacks into ./env/, writes ./activate.sh
source activate.sh                  # activate; no conda/pixi needed on the host
gmx_image --help                    # test biobbs are available
haddock3 --version                  # test haddock3 is available
```

## haddock3

The bioconda `haddock_biobb` package ships **no files** — only a post-link script that
runs `pip install haddock3`. conda/mamba run post-link scripts, pixi deliberately does
not, so a packed env has the `biobb_haddock` wrappers but no `haddock3` binary.
`build_haddock3_wheel.sh` builds that wheel locally (PyPI has sdist only) and
`--inject` embeds it in `environment.sh`; nothing is needed on the cluster side.

Rebuild the wheel when the haddock3 version changes, or if `pixi.lock` moves to a
python other than 3.12 (the wheel is tagged `cp312`). It targets glibc 2.14 via the
conda-forge toolchain, and `-march=x86-64-v3` — override with
`HADDOCK_MARCH=x86-64-v4` for AVX-512 on MN5, at the cost of running nowhere older.
