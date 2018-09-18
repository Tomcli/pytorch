FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         wget \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         openssh-client \
         openssh-server && \
     mkdir -p /var/run/sshd &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN ln -s /opt/conda/bin/python /usr/bin/python

# Install Open MPI with CUDA-aware
RUN mkdir /tmp/openmpi && \
   cd /tmp/openmpi && \
   wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.2.tar.gz && \
   tar zxf openmpi-3.0.2.tar.gz && \
   cd openmpi-3.0.2 && \
   ./configure --enable-orterun-prefix-by-default --with-cuda && \
   make -j $(nproc) all && \
   make install && \
   ldconfig && \
   rm -rf /tmp/openmpi

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
   echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
   echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
   echo '#!/bin/bash' > /usr/local/bin/mpirun && \
   echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
   chmod a+x /usr/local/bin/mpirun

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
   echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
   mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .

RUN git submodule update --init
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

WORKDIR /workspace
RUN chmod -R a+w /workspace
