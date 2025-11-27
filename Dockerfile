FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# We copy requirements.txt first to leverage Docker cache
COPY requirements.txt .
# We install requirements but avoid reinstalling torch if it's already present and compatible
# to prevent overwriting the pre-installed CUDA-optimized version.
RUN pip install --no-cache-dir -r requirements.txt

# Install MambaCD specific dependencies (MMCV series)
# Using mim to install mmcv is recommended as it handles pre-built wheels better
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine==0.10.1 && \
    mim install "mmcv==2.1.0" && \
    pip install --no-cache-dir mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0

# Install selective_scan kernel
# We copy just the kernel directory first
COPY kernels/selective_scan /app/kernels/selective_scan
WORKDIR /app/kernels/selective_scan
# Install the kernel
RUN pip install .

# Copy the rest of the application
WORKDIR /app
COPY . .

# Set PYTHONPATH to include the project root
ENV PYTHONPATH=/app

# Default command
CMD ["/bin/bash"]
