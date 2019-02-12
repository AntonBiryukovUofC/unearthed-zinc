FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt /
COPY setup.py /

RUN pip install -U -r requirements.txt


