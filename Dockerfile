FROM tensorflow/tensorflow:2.4.1-gpu
#FROM tensorflow/tensorflow:2.4.1

WORKDIR /usr/src/app
COPY ./example.py ./
COPY ./PS-Amiloidosis ./PS-Amiloidosis

RUN pip install autokeras