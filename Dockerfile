# This image already includes Python 3.9, JupyterLab and base data science tools + CUDA 11.8 + CuDNN 8.6
FROM quay.io/opendatahub-contrib/workbench-images:cuda-jupyter-datascience-c9s-py39_2023b_latest

# Switch to root to be able to install OS packages
USER 0

# Install the CUDA toolkit. The CUDA repos were already defined in the base image
RUN yum -y install cuda-toolkit-11-8 && \
    yum -y clean all  --enablerepo='*'

# Set this flag so that libraries can find the location of CUDA
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Switch to default user to install Python packages
USER 1001

# Install PyTorch with pip to set the version that matches our CUDA version (does not work with Pipenv).
# Also installing torchvision and torch audio as an example
RUN echo "Installing softwares and packages" && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other NVidia packages: CuDF and CuML
# We have first to remove Elyra, KFP and Streamlit (protobuf version not compatible)
RUN pip uninstall -y kfp kfp-pipeline-spec elyra streamlit jupyterlab-git && \
    # Put back JupyterLab Git
    pip install --no-cache-dir --upgrade jupyterlab==3.5.3 jupyterlab-git==0.41.0 && \
    # Finally install CuDF and CuML
    pip install --no-cache-dir cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com

# Fix permissions to support pip in Openshift environments
RUN chmod -R g+w /opt/app-root/lib/python3.9/site-packages && \
    fix-permissions /opt/app-root -P

# Entrypoint, scripts and other ENV vars already set in base image