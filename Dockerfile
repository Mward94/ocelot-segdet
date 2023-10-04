FROM anibali/pytorch:1.11.0-cuda11.5-ubuntu20.04

# Set environment variable and make required directories for Ocelot
ENV DEBIAN_FRONTEND=noninteractive
RUN sudo mkdir -p /opt/app /input /output \
    && sudo chown user:user /opt /opt/app /input /output

# Set working directory
WORKDIR /opt/app

# Setup timezone (for tzdata dependency install)
ENV TZ=Australia/Melbourne
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install OpenCV dependencies
# * OpenCV: libgl1 libsm6 libxext6 libglib2.0-0
RUN sudo apt-get update \
 && sudo apt-get install -y \
    libgl1 libsm6 libxext6 libglib2.0-0 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install project requirements.
COPY --chown=user:user environment.yml /opt/environment.yml
RUN mamba env update -n base -f /opt/environment.yml \
 && mamba clean -ya

# Copy files across.
COPY --chown=user:user . /opt/app

ENTRYPOINT [ "python", "-m", "process" ]
