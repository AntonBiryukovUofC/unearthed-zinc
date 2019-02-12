FROM tensorflow/tensorflow:latest-gpu

COPY requirements.txt /
COPY setup.py /

RUN pip install -U -r requirements.txt


