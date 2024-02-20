# Build from a base image. Note that we are building SAM from a newer PyTorch image than the one used in the original Dockerfile.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda/

# Since Dockerfile is now in vlm4sgg/, adjust the COPY command to include the Grounded-Segment-Anything subdirectory.
# This will copy the entire vlm4sgg/ directory content into /usr/src/app/, including Grounded-Segment-Anything
RUN mkdir -p /usr/src/app/vlm4sgg
COPY . /usr/src/app/vlm4sgg

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano \
    vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

# Adjust WORKDIR to reflect the new path structure
WORKDIR /usr/src/app/vlm4sgg/Grounded-Segment-Anything
RUN python -m pip install --no-cache-dir -e segment_anything

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel
RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

# Assuming GroundingDINO has a setup.py for compiling its C++/CUDA extensions
WORKDIR /usr/src/app/vlm4sgg/Grounded-Segment-Anything/GroundingDINO
RUN python setup.py build_ext --inplace

# Back to the vlm4sgg level for further installations and operations
WORKDIR /usr/src/app/vlm4sgg
# Resolve the missing gcc package for the installation of pycocotools
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

# Installing CLIP-Count requirements
WORKDIR /usr/src/app/vlm4sgg/CLIP_Count
RUN pip install -r requirements.txt  \
    && pip install ftfy regex tqdm imgaug einops pytorch-lightning  \
    && pip install git+https://github.com/openai/CLIP.git

WORKDIR /usr/src/app/vlm4sgg/

# Now, the requirements.txt file is within the build context, so no need to step out.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

