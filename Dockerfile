FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV TZ=Etc/UTC
WORKDIR /workspace

# 1. Cài đặt hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg xvfb python3.8 python3.8-venv python3.8-dev \
    python3-pip openjdk-11-jdk-headless libgl1-mesa-glx libxrender1 \
    libsm6 libxext6 libgl1 libglu1-mesa libxi-dev libxrandr-dev \
    libxxf86vm-dev libxinerama-dev libxcursor-dev \
    && apt-get clean

# 2. Set Python 3.8 làm default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 3. Tạo virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# 4. Cài torch trước (vì phiên bản torch yêu cầu riêng)
RUN pip install --upgrade pip
RUN pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113

# 5. Clone minerl và cài đặt
RUN git clone -b v1.0.2 https://github.com/minerllabs/minerl.git
WORKDIR /workspace/minerl

# Sửa lỗi về yêu cầu Java path
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

RUN pip install -e .

# 6. Cài các thư viện phụ trợ
RUN pip install opencv-python imageio gym[atari] scikit-learn tqdm pandas matplotlib seaborn

# 7. Tạo thư mục chứa dataset
RUN mkdir -p /workspace/datasets

# 8. Đặt lại working dir
WORKDIR /workspace

# 9. Chạy mặc định là bash
CMD ["/bin/bash"]
