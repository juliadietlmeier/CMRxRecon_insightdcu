## Start from this Docker image
## for the version, we recommend the version xx.xx less than 22.02
##FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive
## Set workdir in Docker Container
# set default workdir in your docker container
# In other words your scripts will run from this directory
WORKDIR /workdir

## Copy all your files of the current folder into Docker Container
COPY ./ /workdir

RUN chmod a+x /workdir/inference.py

## Install requirements
RUN pip3 install -r requirements.txt
RUN pip3 install opencv-python-headless
##RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

## Make Docker container executable
ENTRYPOINT ["/opt/conda/bin/python", "inference.py"]
##ENTRYPOINT ["nvidia-smi"]
