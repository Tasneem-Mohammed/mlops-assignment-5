FROM python:3.10-slim
ARG RUN_ID
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN echo "RUN_ID: ${RUN_ID}"
CMD ["python", "-c", "print('Model container is running')"]
