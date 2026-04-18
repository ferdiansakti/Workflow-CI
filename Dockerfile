FROM python:3.10-slim

WORKDIR /app

COPY MLProject /app/MLProject

RUN pip install pandas scikit-learn mlflow

CMD ["python", "MLProject/modelling.py"]