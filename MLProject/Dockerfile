FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install pandas scikit-learn mlflow

CMD ["python", "modelling.py"]