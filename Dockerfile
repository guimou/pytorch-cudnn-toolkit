# Start  from CentOS Stream 9 + Python 3.9, ref: https://github.com/sclorg/s2i-python-container
FROM quay.io/sclorg/python-39-c9s:latest

ARG RELEASE
ARG DATE

LABEL name="dsrl-custom:${RELEASE}_${DATE}" \
    summary="Data Science Research Lab image" \
    description="Centos Stream 9-Python 3.9-custom bundle of packages" \
    io.k8s.description="Centos Stream 9-Python 3.9-custom bundle of packages" \
    io.k8s.display-name="Centos Stream 9-Python 3.9-custom bundle of packages" \
    authoritative-source-url="https://github.com/guimou/pytorch-cudnn-toolkit" \
    io.openshift.build.commit.ref="${RELEASE}" \
    io.openshift.build.source-location="https://github.com/guimou/pytorch-cudnn-toolkit"



# Switch to root to be able to install OS packages
USER 0

##################################
# CUDA Layer: CUDA+CuDNN+Toolkit #
##################################

# 1. CUDA Base
# ------------
ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.8 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 brand=titan,driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511 brand=unknown,driver>=515,driver<516 brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=515,driver<516 brand=geforce,driver>=515,driver<516 brand=geforcertx,driver>=515,driver<516 brand=quadro,driver>=515,driver<516 brand=quadrortx,driver>=515,driver<516 brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=515,driver<516"
ENV NV_CUDA_CUDART_VERSION 11.8.89-1

COPY cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

ENV CUDA_VERSION 11.8.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-11-8-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-11-8 \
    && ln -s cuda-11.8 /usr/local/cuda \
    && yum clean all --enablerepo='*' \
    && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 2. CUDA Runtime
# ---------------
ENV NV_CUDA_LIB_VERSION 11.8.0-1
ENV NV_NVTX_VERSION 11.8.86-1
ENV NV_LIBNPP_VERSION 11.8.0.86-1
ENV NV_LIBNPP_PACKAGE libnpp-11-8-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION 11.11.3.6-1
ENV NV_LIBNCCL_PACKAGE_NAME libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION 2.15.5-1
ENV NV_LIBNCCL_VERSION 2.15.5
ENV NCCL_VERSION 2.15.5
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.8

RUN yum install -y \
    cuda-libraries-11-8-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-11-8-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-11-8-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all --enablerepo='*' \
    && rm -rf /var/cache/yum/*

# 3. CuDNN
# --------
ENV NV_CUDNN_VERSION 8.6.0.163-1
ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda11.8

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    && yum clean all --enablerepo='*' \
    && rm -rf /var/cache/yum/*

# 4. CUDA Devel (Optional)
# ------------------------
#ENV NV_CUDA_LIB_VERSION 11.8.0-1
#ENV NV_NVPROF_VERSION 11.8.87-1
#ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-11-8-${NV_NVPROF_VERSION}
#ENV NV_CUDA_CUDART_DEV_VERSION 11.8.89-1
#ENV NV_NVML_DEV_VERSION 11.8.86-1
#ENV NV_LIBCUBLAS_DEV_VERSION 11.11.3.6-1
#ENV NV_LIBNPP_DEV_VERSION 11.8.0.86-1
#ENV NV_LIBNPP_DEV_PACKAGE libnpp-devel-11-8-${NV_LIBNPP_DEV_VERSION}
#ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-devel
#ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.15.5-1
#ENV NCCL_VERSION 2.15.5
#ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.8

#RUN yum install -y \
#    make \
#    cuda-command-line-tools-11-8-${NV_CUDA_LIB_VERSION} \
#    cuda-libraries-devel-11-8-${NV_CUDA_LIB_VERSION} \
#    cuda-minimal-build-11-8-${NV_CUDA_LIB_VERSION} \
#    cuda-cudart-devel-11-8-${NV_CUDA_CUDART_DEV_VERSION} \
#    ${NV_NVPROF_DEV_PACKAGE} \
#    cuda-nvml-devel-11-8-${NV_NVML_DEV_VERSION} \
#    libcublas-devel-11-8-${NV_LIBCUBLAS_DEV_VERSION} \
#    ${NV_LIBNPP_DEV_PACKAGE} \
#    ${NV_LIBNCCL_DEV_PACKAGE} \
#    && yum clean all  --enablerepo='*' \
#    && rm -rf /var/cache/yum/*

#ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# 5. CuDNN Devel (Optional)
# ------------------------
#ENV NV_CUDNN_VERSION 8.6.0.163-1
#ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda11.8
#ENV NV_CUDNN_PACKAGE_DEV libcudnn8-devel-${NV_CUDNN_VERSION}.cuda11.8

#LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

#RUN yum install -y \
#    ${NV_CUDNN_PACKAGE} \
#    ${NV_CUDNN_PACKAGE_DEV} \
#    && yum clean all  --enablerepo='*' \
#    && rm -rf /var/cache/yum/*


# 6. CUDA Toolkit
# ---------------
# Install the CUDA toolkit. The CUDA repos were already set
RUN yum -y install cuda-toolkit-11-8 && \
    yum -y clean all  --enablerepo='*' \
    && rm -rf /var/cache/yum/*

# Set this flag so that libraries can find the location of CUDA
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda



##################################
# OS Packages and libraries      #
##################################

# Copy packages list
COPY os-packages.txt ./

RUN yum install -y yum-utils && \
    yum-config-manager --enable crb && \
    yum install -y https://download.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    yum install -y --setopt=tsflags=nodocs $(cat os-packages.txt) && \
    yum -y clean all --enablerepo='*' \
    && rm -rf /var/cache/yum/*


