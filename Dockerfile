FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip3 install --upgrade pip

WORKDIR /workdir
COPY requirements.txt /workdir/

RUN pip3 install -r requirements.txt

COPY . /workdir/
