# Proyecto de detección de implementos de seguridad en tiempo real como servicio a través de un computador industrial.

## Crear y Subir una Imagen a Dockerhub usando Docker Compose

### 1. Crear la Imagen con `docker compose`

Primero, asegúrate de tener un archivo `docker-compose.yml` configurado correctamente con la definición de tu servicio e imagen. Para crear la imagen, usa el siguiente comando:

```bash
docker compose build
```
### 2. Etiquetar la Imagen
Antes de subir la imagen a Docker Hub, debes etiquetarla con tu nombre de usuario de Docker Hub y el nombre del repositorio. Por ejemplo, si tu nombre de usuario es usuario y el nombre de la imagen es miimagen, harías lo siguiente:

```bash
docker tag nombre_de_la_imagen:tag usuario/miimagen:tag
```
Por ejemplo, si el nombre de la imagen es myapp, y la quieres subir como usuario/myapp:latest, ejecuta:

```bash
docker tag myapp:latest usuario/myapp:latest
```

### 3. Iniciar Sesión en Docker Hub
Inicia sesión en Docker Hub desde la línea de comandos:

```bash
docker login
```
Luego de ingresar credenciales en nuestra cuenta subimos la imagen.

### Subir la Imagen a Docker Hub
Subir imagen etiquetada a Docker Hub:

```bash
docker push usuario/miimagen:tag
```

### 4. Hacer Pull de la imagen desde computador industrial ( EPC1502 ), previamente realizando docker login también:

```bash
docker pull usuario/myapp:latest
```
### 5. Ejecutar imagen en contenedor (equivalente con balena, reemplazando alias "docker" por "balena-engine"):

```bash
docker run usuario/miimagen:tag 
```

Sin embargo debemos tener en consideración que el servicio de transmisión de frames debe estar habilitado. En nuestro caso particular lo hacemos de la siguiente forma:

## Configuración y Ejecución del Servidor RTSP

### 1. Crear el Archivo de Configuración para el Servidor RTSP

Crea un archivo de configuración llamado `rtsp-simple-server.yml` con el siguiente contenido:

```yaml
protocols: [tcp]
paths:
  all:
    source: publisher
```

### 2. Iniciar servicio RTSP como un contenedor Docker:

```bash
docker run --rm -it -v $PWD/rtsp-simple-server.yml:/rtsp-simple-server.yml -p 8554:8554 aler9/rtsp-simple-server:v1.3.0
```

### 3. Transmitir un Archivo de Video (frames) al Servidor RTSP con un dispositivo, camara en especifico (Azure Kinect DK):

```bash
ffmpeg -f dshow -rtbufsize 4M -video_size 1920x1080 -i video="Azure Kinect 4K Camera" -c:v libx264 -f rtsp -rtsp_transport tcp rtsp://localhost:8554/stream
````

## Visualizar detección:

Finalmente en la url designada en el archivo python: app.py -> 127.0.0.1:5000/video_feed se verán reflejados la detección en tiempo real.
