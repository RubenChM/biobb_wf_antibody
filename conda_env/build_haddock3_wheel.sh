#!/usr/bin/env bash
# Build a portable haddock3 wheel to ship next to the pixi-pack environment.
#
# Why this exists: the bioconda `haddock_biobb` package contains no files, only a
# post-link script that runs `pip install haddock3`. conda/mamba execute post-link
# scripts, pixi (and pixi-unpack) deliberately do not, so a pixi-packed env has the
# biobb_haddock wrappers but no haddock3 binary. We build the wheel once, here, and
# install it into the unpacked env on the cluster.
#
# Two portability traps are handled:
#   * haddock3 is only published as an sdist, and building it with the distro gcc
#     links against the host glibc (2.38+ on Ubuntu 24.04) -> fails on the cluster.
#     We build with the conda-forge toolchain on a 2.17 sysroot instead.
#   * setup.py hardcodes `-march=native` for fast-rmsdmatrix, i.e. the *build* CPU.
#     A gcc/g++ shim rewrites it to $HADDOCK_MARCH (default x86-64-v3 / AVX2).
#
# Usage: ./build_haddock3_wheel.sh [haddock3_version] [python_version]
set -euo pipefail

HADDOCK3_VERSION="${1:-2025.11.0}"
PY_VERSION="${2:-3.12}"          # must match the python in pixi.lock
HADDOCK_MARCH="${HADDOCK_MARCH:-x86-64-v3}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> creating conda-forge toolchain env (sysroot 2.17, python $PY_VERSION)"
conda create -y -p "$WORK/buildenv" -c conda-forge \
  "python=$PY_VERSION" pip setuptools wheel \
  gcc_linux-64 gxx_linux-64 "sysroot_linux-64=2.17" >/dev/null

mkdir -p "$WORK/shims"
cat > "$WORK/shims/gcc" <<'EOF'
#!/usr/bin/env bash
args=()
for a in "$@"; do
  if [ "$a" = "-march=native" ]; then args+=("-march=${HADDOCK_MARCH}"); else args+=("$a"); fi
done
exec "$CONDA_GCC" "${args[@]}"
EOF
sed 's|\$CONDA_GCC|$CONDA_GXX|' "$WORK/shims/gcc" > "$WORK/shims/g++"
chmod +x "$WORK/shims/gcc" "$WORK/shims/g++"

export CONDA_GCC="$WORK/buildenv/bin/x86_64-conda-linux-gnu-gcc"
export CONDA_GXX="$WORK/buildenv/bin/x86_64-conda-linux-gnu-g++"
export HADDOCK_MARCH
export PATH="$WORK/shims:$WORK/buildenv/bin:$PATH"
export CC="$WORK/shims/gcc" CXX="$WORK/shims/g++"

echo ">> building haddock3 $HADDOCK3_VERSION wheel (-march=$HADDOCK_MARCH)"
# --no-deps: the conda env already provides haddock3's python dependencies,
# exactly as the bioconda post-link script assumes.
"$WORK/buildenv/bin/python" -m pip wheel "haddock3==$HADDOCK3_VERSION" \
  --no-deps --no-build-isolation --no-cache-dir -w "$HERE"

WHEEL="$(ls -t "$HERE"/haddock3-"$HADDOCK3_VERSION"-*.whl | head -1)"
echo ">> $WHEEL"

echo ">> checking portability of the compiled bits"
( cd "$WORK" && unzip -q -o "$WHEEL" 'haddock/bin/*' -d check
  for f in check/haddock/bin/contact_fcc check/haddock/bin/fast-rmsdmatrix; do
    glibc="$(objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
    avx512="$(objdump -d "$f" | grep -cE '%zmm|vpermt2|kmov' || true)"
    printf '   %-24s max %s, avx512 insns: %s\n' "$(basename "$f")" "${glibc:-none}" "$avx512"
  done )
