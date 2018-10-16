FROM python:3.6

WORKDIR /usr/src/app

ENV PYTHONPATH="/usr/src/app:${PYTHONPATH}"