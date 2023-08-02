# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY . .

RUN pip3 install -r requirements/requirements.txt

CMD [ "python3", "titanic_model/server.py", "--host=0.0.0.0"]
