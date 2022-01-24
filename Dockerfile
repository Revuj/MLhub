FROM python:3.10-slim



RUN adduser mluser --gecos "" --disabled-password
USER mluser
WORKDIR /home/mluser
ENV PATH "$PATH:/home/mluser/.local/bin"
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install setuptools wheel
RUN pip3 install -r ~/requirements.txt
COPY src src
