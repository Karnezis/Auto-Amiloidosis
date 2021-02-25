FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /usr/src/app
COPY ./example.py ./

RUN pip install autokeras