# Usamos la versión más estable y ligera
FROM python:3.12-slim

WORKDIR /app

# Instalamos dependencias (Capa de caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Creamos la estructura de directorios necesaria
RUN mkdir -p models

# Copiamos el código de la API
COPY main.py .

# Copiamos el directorio de modelos (contiene el Pipeline pkl de la ultima versión)
COPY models/ ./models/

# Exponemos el puerto de la IA
EXPOSE 8000

# Iniciamos el servicio directamente
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
