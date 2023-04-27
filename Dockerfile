FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && \
  apt-get install -y python3 python3-pip && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*$

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libopencv-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libatlas-base-dev \
        gfortran \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        vim \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC

# RUN apt-get update && apt-get install -y \
#     libopencv-core-dev \
#     libopencv-highgui-dev \
#     libopencv-imgproc-dev \
#     libopencv-videoio-dev

# RUN echo "export DISPLAY=$DISPLAY" >> /etc/environment

COPY . .

EXPOSE 8080

CMD nvidia-smi

CMD ["python3", "yolo_slowfast.py", "--input=demo2.mp4", "--device=cpu", "--show"]