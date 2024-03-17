FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip nvidia-opencl-dev clinfo nvidia-opencl-icd-384

RUN pip install "arkitekt[all]"

RUN pip install pyclesperanto-prototype

RUN mkdir /app
WORKDIR /app
COPY .arkitekt /app/.arkitekt
COPY app.py /app/app.py
