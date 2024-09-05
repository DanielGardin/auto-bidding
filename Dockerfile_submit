FROM competition-hub-registry.cn-beijing.cr.aliyuncs.com/alimama-competition/bidding-results:base

# Setting root and copying files
WORKDIR /root/biddingTrainEnv
COPY .. .


# Many boring steps to install python3.11 and pip
RUN apt-get update
RUN apt-get install -y wget 

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.11
RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/local/bin/python3.9 1
RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.11 2
RUN update-alternatives --set python3 /usr/bin/python3.11
RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python

# Install requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the evaluation script
CMD ["python3", "./run/run_evaluate.py"]

ENV PYTHONPATH="/root/biddingTrainEnv:${PYTHONPATH}"