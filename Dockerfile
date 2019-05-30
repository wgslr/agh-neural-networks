FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

WORKDIR /workdir
COPY requirements.txt /workdir/

RUN pip3 install -r requirements.txt

COPY . /workdir/
