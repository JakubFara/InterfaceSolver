FROM quay.io/fenicsproject/dev:latest
#FROM python:3.9.2
USER root

WORKDIR /home/interfacesolver
ADD ./src /home/interfacesolver/src
ADD ./setup.py /home/interfacesolver
ADD ./README.md /home/interfacesolver
ADD ./examples /home/interfacesolver/examples
RUN ls
RUN pip3 install .