# Start FastAPI via Uvicorn targeting model_service_app.py
CMD ["uvicorn", "src.model_service_app:app", "--host", "0.0.0.0", "--port", "8000"]