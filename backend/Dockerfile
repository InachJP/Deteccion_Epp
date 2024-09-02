# Usa una imagen base de Python
FROM python:3.9-slim

# Actualiza el sistema y instala dependencias
RUN apt-get update && \
    apt-get install -y gcc g++ make libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia los archivos necesarios al contenedor
COPY backend/requirements.txt backend/requirements.txt
COPY backend/app.py backend/app.py
COPY backend/yolo_nas_model.pth backend/yolo_nas_model.pth

# Instala las dependencias
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expone el puerto en el que se ejecutará la aplicación
EXPOSE 5000

# Define el comando para ejecutar la aplicación
CMD ["python", "backend/app.py"]
