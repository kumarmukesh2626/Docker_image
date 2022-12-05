From python:3.7

COPY . /app

WORKDIR /app

MAINTAINER MUKESH 

RUN pip install --upgrade pip

RUN apt-get update

RUN apt install -y libgl1-mesa-glx

RUN pip install -r /app/yolov5/requirements.txt

CMD ["python","yolov5.py"]


