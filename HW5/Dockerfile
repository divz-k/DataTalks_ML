FROM agrigorev/zoomcamp-model:2025

WORKDIR /app
COPY app.py .
COPY pipeline_v1.bin .
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
