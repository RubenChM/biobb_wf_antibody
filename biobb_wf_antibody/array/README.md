# pixi-pack instructions
``` shell
cd ../conda_env
pixi lock
pixi-pack --create-executable
scp environment.sh user@<cluster>:/path/to/biobb_wf_antibody/
# On cluster, run:
./environment.sh                    # unpacks into ./env/, writes ./activate.sh
source activate.sh                  # activate; no conda/pixi needed on the host
gmx_image --help                    # test biobbs are available
```
