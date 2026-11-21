FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-api-lock-linux.txt .
RUN pip install --no-cache-dir -r requirements-api-lock-linux.txt
COPY app.py model.joblib model_context.json ./
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
