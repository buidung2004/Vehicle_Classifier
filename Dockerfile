FROM python:3.8.10

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 5000
RUN  gdown --id 1-ANbcgaKUDkmsDrS96_km7Jk2aF0-Xl3 
CMD ["python", "app.py"]
