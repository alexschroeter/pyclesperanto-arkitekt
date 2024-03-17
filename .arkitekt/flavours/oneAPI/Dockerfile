FROM intel/oneapi-basekit:devel-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip intel-oneapi-runtime-opencl clinfo intel-opencl-icd

RUN pip install "arkitekt[all]"

RUN pip install pyclesperanto-prototype

RUN mkdir /app
WORKDIR /app
COPY .arkitekt /app/.arkitekt
COPY app.py /app/app.py
