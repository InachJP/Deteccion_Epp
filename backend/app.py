from super_gradients.training import models
import cv2
from flask import Flask, Response, stream_with_context
import numpy as np
import random
import torch
import threading
import queue

# Parámetros predefinidos
NUM_CLASSES = 7
MODEL_TYPE = 'yolo_nas_s'
WEIGHT_PATH = '/backend/yolo_nas_model.pth'
SOURCE = ['rtsp://localhost:8554/stream']
CONFIDENCE_THRESHOLD = 0.35

# Inicializar la aplicación Flask
app = Flask(__name__)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Cargar el modelo YOLO-NAS
model = models.get(
    MODEL_TYPE,
    num_classes=NUM_CLASSES,
    checkpoint_path=WEIGHT_PATH
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Obtener los nombres de las clases
class_names = model.predict(np.zeros((1, 1, 3)), conf=CONFIDENCE_THRESHOLD)._images_prediction_lst[0].class_names
print('Class Names:', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# Inicializar la captura de video para todas las fuentes
cap_list = [cv2.VideoCapture(source) for source in SOURCE]

# Buffer para acumular frames
frame_queue = queue.Queue(maxsize=60)
stop_event = threading.Event()

def capture_frames():
    while not stop_event.is_set():
        for cap in cap_list:
            success, img = cap.read()
            if not success:
                print(f'[INFO] Failed to read from source {cap}')
                continue
            if not frame_queue.full():
                frame_queue.put(img)
            else:
                frame_queue.get()
                frame_queue.put(img)

def process_batch(img_list):
    img_rgb_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
    preds = model.predict(img_rgb_list, conf=CONFIDENCE_THRESHOLD)._images_prediction_lst
    return preds

def generate_frames():
    while True:
        if frame_queue.qsize() >= 30:
            batch = [frame_queue.get() for _ in range(30)]
            preds = process_batch(batch)
            
            for id, pred in enumerate(preds):
                dp = pred.prediction
                bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
                for box, cnf, cs in zip(bboxes, confs, labels):
                    plot_one_box(box[:4], batch[id], label=f'{class_names[int(cs)]} {cnf:.3f}', color=colors[cs])
            
            for img in batch:
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            continue

# Definir una ruta en Flask para el stream de video
@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutar la aplicación Flask y el hilo de captura de frames
if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        stop_event.set()
        capture_thread.join()
        for cap in cap_list:
            cap.release()
        cv2.destroyAllWindows()
