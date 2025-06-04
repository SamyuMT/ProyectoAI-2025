FROM python:3.13-slim

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean

# Instalar dependencias de Python
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . /app

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]