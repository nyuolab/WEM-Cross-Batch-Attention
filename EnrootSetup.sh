#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04}"
CONTAINER_NAME="${CONTAINER_NAME:-cuda-python-build}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
USE_VENV="${USE_VENV:-1}"            
APT_NONINTERACTIVE="${APT_NONINTERACTIVE:-1}"  

usage() {
  cat <<EOF
Usage: $(basename "$0") [--base <image>] [--name <container>] [--out <file.sqsh>] [--req <requirements.txt>] [--no-venv]
  --base    NVIDIA container image (default: $BASE_IMAGE)
  --name    Temporary enroot container name (default: $CONTAINER_NAME)
  --req     Path to requirements.txt (default: $REQUIREMENTS)
  --no-venv Disable creating a virtualenv (/opt/venv)
EOF
  exit 1
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE_IMAGE="$2"; shift 2;;
    --name) CONTAINER_NAME="$2"; shift 2;;
    --req)  REQUIREMENTS="$2"; shift 2;;
    --no-venv) USE_VENV=0; shift;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

command -v enroot >/dev/null 2>&1 || { echo "enroot not found in PATH"; exit 1; }
[[ -f "$REQUIREMENTS" ]] || { echo "requirements file not found: $REQUIREMENTS"; exit 1; }

echo "[1/5] Importing base image: $BASE_IMAGE"
BASE_REF="docker://$BASE_IMAGE"
enroot import "$BASE_REF"

BASE_SQSH="$(ls -t *.sqsh | head -n1)"
[[ -f "$BASE_SQSH" ]] || { echo "Failed to locate imported .sqsh"; exit 1; }
echo "    -> Using base squashfs: $BASE_SQSH"

echo "[2/3] Creating container: $CONTAINER_NAME"

# --force to overwrite if it already exists
enroot create --force --name "$CONTAINER_NAME" "$BASE_SQSH"

echo "[3/3] Installing Python, pip, build tools, and your requirements"
MOUNT_REQ_DIR="$(cd "$(dirname "$REQUIREMENTS")" && pwd)"
REQ_BASENAME="$(basename "$REQUIREMENTS")"
echo $MOUNT_REQ_DIR
echo $REQ_BASENAME

read -r -d '' INSIDE_SCRIPT <<'EOS' || true
set -euo pipefail
if [[ "${APT_NONINTERACTIVE:-1}" == "1" ]]; then
  export DEBIAN_FRONTEND=noninteractive
fi

nvcc --version
apt-get update
apt-get install -y --no-install-recommends \
  libbz2-dev python3 python3-venv python3-dev \
  build-essential git ca-certificates curl pkg-config ninja-build

update-alternatives --install /usr/bin/python python /usr/bin/python3 10 || true

if [[ "${USE_VENV:-1}" == "1" ]]; then
  python -m venv /opt/venv
  echo 'export PATH=/opt/venv/bin:$PATH' >/etc/profile.d/venv.sh
  source /opt/venv/bin/activate
fi

pip install --upgrade pip setuptools wheel ninja packaging
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128


mkdir -p /opt/app
cp -f /workspace/"$REQ_BASENAME" /opt/app/requirements.txt

pip install --no-cache-dir -r /opt/app/requirements.txt


# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# MAX_JOBS=32 python setup.py install
# cd ../..

# Clean apt lists to reduce image size
rm -rf /var/lib/apt/lists/*
python --version
pip --version
EOS

enroot start --root --rw \
  --mount "$MOUNT_REQ_DIR:/workspace" \
  --env USE_VENV="$USE_VENV" \
  --env APT_NONINTERACTIVE="$APT_NONINTERACTIVE" \
  "$CONTAINER_NAME" bash -lc "REQ_BASENAME='$REQ_BASENAME' USE_VENV='$USE_VENV' APT_NONINTERACTIVE='$APT_NONINTERACTIVE' bash -s <<'INNERSCRIPT'
$INSIDE_SCRIPT
INNERSCRIPT
"

