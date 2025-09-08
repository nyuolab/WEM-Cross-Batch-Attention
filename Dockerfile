
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev \
      build-essential git ca-certificates curl pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Make `python` point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Create a virtual environment (optional but recommended for cleanliness)
ENV VENV_PATH=/opt/venv
RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip/setuptools/wheel for faster, more reliable builds
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128


# Set workdir and copy only requirements first for better layer caching
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
#     cd flash-attention/hopper && \
#     python setup.py install
# Now bring in the rest of your app (if any)
# COPY . /app

# Default command (adjust as needed)
CMD ["python", "--version"]
