ARG CUDA_VERSION=12.2.0

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV SHELL=/bin/bash

WORKDIR /work/

RUN apt-get update
RUN apt-get install -y \
    curl \
    wget \
    git \
    openssh-client

RUN mkdir -p ~/.ssh

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y \
    build-essential \ 
    python3.11 \
    python3.11-dev \
    libmpc-dev

RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python


COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

RUN git config --global --add safe.directory /work/

EXPOSE 8000