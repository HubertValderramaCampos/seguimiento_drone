#!/usr/bin/env python3
"""
Script 2: Seguimiento PTZ ESTABLE con control PID
- Movimiento suave y predecible
- Sin oscilaciones
- Zona muerta adaptativa
- Control desde base de datos PostgreSQL
"""
import os
import sys
import time
import threading
import signal
from collections import deque

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2
import numpy as np
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
import requests
import base64
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# ============== CONFIGURACIÓN ==============
# Base de datos
DB_HOST = os.getenv("DB_HOST", "178.18.254.186")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "drondb")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345")
DB_CHECK_INTERVAL = 2  # segundos entre chequeos

# Evolution API (WhatsApp)
EVOLUTION_BASE_URL = os.getenv("BASE_URL_EVOLUTION_API", "").strip()
EVOLUTION_API_KEY = os.getenv("API_KEY_EVOLUTION_API", "").strip()
EVOLUTION_INSTANCE = os.getenv("INSTANCE_EVOLUTION_API", "").strip()
NUMERO_WHATSAPP = os.getenv("NUMERO_ENVIAR_SMS", "").strip()
VIDEO_DURACION = 25  # segundos de video a capturar
VIDEO_COOLDOWN = 60  # segundos entre envíos (evitar spam)

# PTZ Camera
IP = "192.168.101.19"
RTSP_URL = f"rtsp://{IP}:554/user=admin&password=&channel=1&stream=0.sdp"
ONVIF_PORT = 8899
ONVIF_USER = "admin"
ONVIF_PASS = ""

# Modelo
MODELO_YOLO = "objetivo_modelo.pt"
USAR_PERSONA_GENERAL = False

# PTZ - EJES
INVERTIR_PAN = True
INVERTIR_TILT = True

# Control PID - MÁXIMA VELOCIDAD Y RESPUESTA
KP = 0.18          # Proporcional (muy alto para respuesta inmediata)
KI = 0.003         # Integral (corrección rápida)
KD = 0.07          # Derivativo (mantiene estabilidad mínima)

# Zonas
ZONA_MUERTA = 0.04       # 4% - zona muy pequeña
ZONA_PRECISION = 0.15    # 15% - zona de desaceleración reducida
VEL_MIN = 0.04           # Velocidad mínima aumentada
VEL_MAX = 0.55           # Velocidad máxima muy alta (antes 0.40)

# Suavizado - REDUCIDO para respuesta directa
FILTRO_POSICION = 0.6    # Menos suavizado (más directo)
FILTRO_VELOCIDAD = 0.7   # Cambios rápidos de velocidad
HISTORIAL_SIZE = 3       # Solo 3 frames (mínimo lag)

# Performance
SKIP_FRAMES = 1          # Procesar cada frame (antes 2, más rápido)
CONF_THRESHOLD = 0.45
BUFFER_SIZE = 60

# Visualización
MOSTRAR_VIDEO = True
# ==========================================


class ControladorDB:
    """Monitorea la base de datos para controlar encendido/apagado del sistema"""
    def __init__(self):
        self.conn = None
        self.running = True
        self.encendido = False
        self.conectar()

    def conectar(self):
        """Establece conexión con PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print(f"[DB] Conectado a PostgreSQL en {DB_HOST}:{DB_PORT}/{DB_NAME}")
            return True
        except Exception as e:
            print(f"[DB ERROR] No se pudo conectar: {e}")
            return False

    def verificar_encendido(self):
        """Consulta el estado de 'encendido' desde control_dron"""
        if not self.conn or self.conn.closed:
            if not self.conectar():
                return False

        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT encendido FROM control_dron WHERE id = 1")
                result = cur.fetchone()
                if result:
                    self.encendido = result[0]
                    return self.encendido
                return False
        except Exception as e:
            print(f"[DB ERROR] Error al consultar: {e}")
            self.conn = None
            return False

    def apagar_sistema(self):
        """Actualiza encendido = FALSE en la base de datos"""
        if not self.conn or self.conn.closed:
            if not self.conectar():
                return False

        try:
            with self.conn.cursor() as cur:
                cur.execute("UPDATE control_dron SET encendido = FALSE WHERE id = 1")
                self.conn.commit()
                print("[DB] Sistema marcado como apagado en BD")
                return True
        except Exception as e:
            print(f"[DB ERROR] Error al actualizar: {e}")
            self.conn.rollback()
            return False

    def cerrar(self):
        """Cierra la conexión"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("[DB] Conexión cerrada")


class VideoGrabador:
    """Graba video de 5 segundos y envía por WhatsApp usando Evolution API"""
    def __init__(self):
        self.grabando = False
        self.ultimo_envio = 0
        self.video_path = None

    def puede_enviar(self):
        """Verifica si ha pasado suficiente tiempo desde el último envío"""
        return (time.time() - self.ultimo_envio) >= VIDEO_COOLDOWN

    def grabar_video(self, stream):
        """Graba VIDEO_DURACION segundos de video desde el stream"""
        if self.grabando:
            return None

        self.grabando = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"sospechoso_{timestamp}.mp4"
        video_path = os.path.join(os.getcwd(), video_filename)

        print(f"[VIDEO] Iniciando grabación de {VIDEO_DURACION}s...")

        try:
            # Obtener dimensiones del frame
            _, sample_frame = stream.get_latest()
            if sample_frame is None:
                print("[VIDEO ERROR] No hay frames disponibles")
                self.grabando = False
                return None

            h, w = sample_frame.shape[:2]

            # Reducir resolución para disminuir tamaño del archivo
            # Reducir a 50% del tamaño original (máximo 640px de ancho)
            scale = 0.5 if w > 640 else 1.0
            new_w = int(w * scale)
            new_h = int(h * scale)

            fps = 15  # Reducir FPS para menor tamaño

            # Configurar codec mp4v (compatible con contenedor MP4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (new_w, new_h))

            if not out.isOpened():
                print("[VIDEO ERROR] No se pudo crear el archivo de video")
                self.grabando = False
                return None

            # Grabar durante VIDEO_DURACION segundos
            frames_grabados = 0
            tiempo_inicio = time.time()

            while (time.time() - tiempo_inicio) < VIDEO_DURACION:
                _, frame = stream.get_latest()
                if frame is not None:
                    # Redimensionar frame si es necesario
                    if scale != 1.0:
                        frame = cv2.resize(frame, (new_w, new_h))
                    out.write(frame)
                    frames_grabados += 1
                time.sleep(1.0 / fps)  # Ajustar según FPS configurado

            out.release()

            # Verificar que el archivo se creó correctamente
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"[VIDEO] Grabación completada: {frames_grabados} frames, {os.path.getsize(video_path)} bytes")
                self.video_path = video_path
                self.grabando = False
                return video_path
            else:
                print("[VIDEO ERROR] El archivo de video está vacío o no existe")
                self.grabando = False
                return None

        except Exception as e:
            print(f"[VIDEO ERROR] Error al grabar: {e}")
            if out:
                out.release()
            self.grabando = False
            return None

    def enviar_video_whatsapp(self, video_path):
        """Envía el video por WhatsApp usando Evolution API v1"""
        if not EVOLUTION_BASE_URL or not EVOLUTION_API_KEY or not EVOLUTION_INSTANCE:
            print("[WHATSAPP ERROR] Credenciales de Evolution API no configuradas")
            return False

        if not NUMERO_WHATSAPP:
            print("[WHATSAPP ERROR] Número de WhatsApp no configurado")
            return False

        try:
            # Leer y codificar el video en base64
            with open(video_path, 'rb') as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode('utf-8')

            # Preparar la URL del endpoint
            url = f"{EVOLUTION_BASE_URL}/message/sendMedia/{EVOLUTION_INSTANCE}"

            # Preparar headers
            headers = {
                'Content-Type': 'application/json',
                'apikey': EVOLUTION_API_KEY
            }

            # Preparar el body según formato Evolution API (base64 puro, sin prefijo)
            timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            payload = {
                'number': NUMERO_WHATSAPP,
                'mediatype': 'video',
                'mimetype': 'video/mp4',
                'caption': f"ALERTA: Sospechoso detectado\nFecha: {timestamp_str}\nDuracion: {VIDEO_DURACION}s",
                'media': video_base64,  # Base64 puro sin prefijo data:video/mp4;base64,
                'delay': 1000
            }

            tamaño_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"[WHATSAPP] Enviando video a {NUMERO_WHATSAPP}...")
            print(f"[WHATSAPP] Tamaño: {tamaño_mb:.2f} MB ({os.path.getsize(video_path)} bytes)")
            print(f"[WHATSAPP] Esto puede tardar varios minutos...")

            # Enviar la petición con timeout largo (5 minutos para videos grandes)
            timeout_segundos = 300  # 5 minutos
            print(f"[WHATSAPP] Timeout configurado: {timeout_segundos}s")

            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout_segundos)

                if response.status_code in [200, 201]:
                    print(f"[WHATSAPP] ✓ Video enviado exitosamente")
                    self.ultimo_envio = time.time()

                    # Eliminar el archivo local después de enviarlo
                    try:
                        os.remove(video_path)
                        print(f"[VIDEO] Archivo temporal eliminado: {video_path}")
                    except:
                        pass

                    return True
                else:
                    print(f"[WHATSAPP ERROR] Error al enviar: {response.status_code}")
                    print(f"[WHATSAPP ERROR] Respuesta: {response.text}")
                    return False

            except requests.exceptions.Timeout:
                print(f"[WHATSAPP ERROR] Timeout al enviar video ({timeout_segundos}s)")
                print(f"[WHATSAPP ERROR] El video es muy grande ({tamaño_mb:.2f} MB)")
                return False
            except requests.exceptions.RequestException as e:
                print(f"[WHATSAPP ERROR] Error de conexión: {e}")
                return False

        except Exception as e:
            print(f"[WHATSAPP ERROR] Error al enviar video: {e}")
            import traceback
            traceback.print_exc()
            return False


class ControladorPID:
    """Controlador PID para movimiento suave"""
    def __init__(self, kp, ki, kd, limite_integral=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limite_integral = limite_integral
        
        self.integral = 0
        self.error_anterior = 0
        self.salida_anterior = 0
        self.tiempo_anterior = time.time()
    
    def calcular(self, error):
        ahora = time.time()
        dt = ahora - self.tiempo_anterior
        if dt <= 0:
            dt = 0.033  # ~30fps
        
        # Proporcional
        P = self.kp * error
        
        # Integral con anti-windup
        self.integral += error * dt
        self.integral = max(-self.limite_integral, min(self.limite_integral, self.integral))
        I = self.ki * self.integral
        
        # Derivativo
        derivada = (error - self.error_anterior) / dt if dt > 0 else 0
        D = self.kd * derivada
        
        # Salida total
        salida = P + I + D
        
        # Filtro de salida para suavizar
        salida = self.salida_anterior * (1 - FILTRO_VELOCIDAD) + salida * FILTRO_VELOCIDAD
        
        # Guardar estado
        self.error_anterior = error
        self.tiempo_anterior = ahora
        self.salida_anterior = salida
        
        return salida
    
    def reset(self):
        self.integral = 0
        self.error_anterior = 0
        self.salida_anterior = 0


class PTZController:
    """Control PTZ con rate limiting"""
    def __init__(self, ip, port, user, password):
        from onvif import ONVIFCamera
        self.cam = ONVIFCamera(ip, port, user, password)
        self.ptz = self.cam.create_ptz_service()
        self.token = self.cam.create_media_service().GetProfiles()[0].token
        self.last_move = time.time()
        self.min_interval = 0.05  # 50ms entre comandos (más rápido, antes 80ms)
        self.lock = threading.Lock()
        self.vel_actual = (0, 0)
    
    def mover(self, pan=0, tilt=0):
        with self.lock:
            now = time.time()
            if now - self.last_move < self.min_interval:
                return
            
            # Limitar velocidad
            pan = max(-VEL_MAX, min(VEL_MAX, pan))
            tilt = max(-VEL_MAX, min(VEL_MAX, tilt))
            
            # Ignorar movimientos muy pequeños
            if abs(pan) < VEL_MIN:
                pan = 0
            if abs(tilt) < VEL_MIN:
                tilt = 0
            
            # Solo enviar si cambió significativamente
            if abs(pan - self.vel_actual[0]) < 0.01 and abs(tilt - self.vel_actual[1]) < 0.01:
                return
            
            try:
                req = self.ptz.create_type('ContinuousMove')
                req.ProfileToken = self.token
                req.Velocity = {
                    'PanTilt': {'x': pan, 'y': tilt},
                    'Zoom': {'x': 0}
                }
                self.ptz.ContinuousMove(req)
                self.last_move = now
                self.vel_actual = (pan, tilt)
            except:
                pass
    
    def parar(self):
        with self.lock:
            if self.vel_actual == (0, 0):
                return
            try:
                req = self.ptz.create_type('Stop')
                req.ProfileToken = self.token
                req.PanTilt = True
                req.Zoom = True
                self.ptz.Stop(req)
                self.vel_actual = (0, 0)
            except:
                pass


class StreamCapture:
    """Captura de stream con buffer circular"""
    def __init__(self, url, buffer_size=60):
        self.url = url
        self.buffer = deque(maxlen=buffer_size)
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.lock = threading.Lock()
        
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.buffer.append((time.time(), frame))
                        self.frame_count += 1
                        
                        now = time.time()
                        if now - self.last_fps_time >= 1.0:
                            self.fps = self.frame_count
                            self.frame_count = 0
                            self.last_fps_time = now
                else:
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            except:
                time.sleep(1)
    
    def get_latest(self):
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
        return None, None
    
    def get_fps(self):
        return self.fps
    
    def release(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass


class Detector:
    """Detector YOLO"""
    def __init__(self, modelo_path, usar_general=False):
        from ultralytics import YOLO
        
        self.usar_general = usar_general
        
        if usar_general or not os.path.exists(modelo_path):
            print(f"[INFO] Usando modelo general YOLOv8")
            self.model = YOLO("yolov8n.pt")
            self.target_class = 0
        else:
            print(f"[INFO] Usando modelo personalizado: {modelo_path}")
            self.model = YOLO(modelo_path)
            self.target_class = 0
        
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[OK] Detector inicializado")
    
    def detectar(self, frame):
        results = self.model(frame, verbose=False, conf=CONF_THRESHOLD)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            if self.usar_general:
                mask = boxes.cls == self.target_class
                if not mask.any():
                    return None
                boxes = boxes[mask]
            
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            idx = areas.argmax()
            
            box = boxes.xyxy[idx].cpu().numpy()
            conf = float(boxes.conf[idx].cpu().numpy())
            
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            return (x_center, y_center, conf, box)
        
        return None


class FiltroKalmanSimple:
    """Filtro simple para suavizar posición"""
    def __init__(self, size=5):
        self.historial_x = deque(maxlen=size)
        self.historial_y = deque(maxlen=size)
        self.pos_filtrada = (0.5, 0.5)
    
    def actualizar(self, x, y):
        self.historial_x.append(x)
        self.historial_y.append(y)
        
        # Promedio ponderado (más peso a recientes)
        if len(self.historial_x) > 0:
            pesos = np.exp(np.linspace(-1, 0, len(self.historial_x)))
            pesos /= pesos.sum()
            
            x_filtrado = np.average(list(self.historial_x), weights=pesos)
            y_filtrado = np.average(list(self.historial_y), weights=pesos)
            
            # Suavizado adicional
            self.pos_filtrada = (
                self.pos_filtrada[0] * (1 - FILTRO_POSICION) + x_filtrado * FILTRO_POSICION,
                self.pos_filtrada[1] * (1 - FILTRO_POSICION) + y_filtrado * FILTRO_POSICION
            )
        
        return self.pos_filtrada
    
    def reset(self):
        self.historial_x.clear()
        self.historial_y.clear()
        self.pos_filtrada = (0.5, 0.5)


class Visualizador:
    """Visualización en thread separado"""
    def __init__(self):
        self.frame_actual = None
        self.info = {}
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()
    
    def actualizar(self, frame, info):
        with self.lock:
            self.frame_actual = frame.copy()
            self.info = info.copy()
    
    def _display_loop(self):
        cv2.namedWindow("SISTEMA DE VIGILANCIA PTZ", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SISTEMA DE VIGILANCIA PTZ", 1280, 720)

        while self.running:
            with self.lock:
                if self.frame_actual is not None:
                    frame = self.frame_actual.copy()
                    info = self.info.copy()
                else:
                    frame = None
                    info = {}

            if frame is not None:
                h, w = frame.shape[:2]

                # Bbox con diseño moderno
                if 'bbox' in info and info['bbox'] is not None:
                    box = info['bbox'].astype(int)
                    conf = info.get('conf', 0)

                    # Rectángulo principal (rojo brillante para sospechoso)
                    color_box = (0, 0, 255)  # Rojo en BGR
                    thickness = 3
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_box, thickness)

                    # Esquinas modernas (estilo high-tech)
                    corner_length = 25
                    corner_thick = 4
                    # Esquina superior izquierda
                    cv2.line(frame, (box[0], box[1]), (box[0] + corner_length, box[1]), color_box, corner_thick)
                    cv2.line(frame, (box[0], box[1]), (box[0], box[1] + corner_length), color_box, corner_thick)
                    # Esquina superior derecha
                    cv2.line(frame, (box[2], box[1]), (box[2] - corner_length, box[1]), color_box, corner_thick)
                    cv2.line(frame, (box[2], box[1]), (box[2], box[1] + corner_length), color_box, corner_thick)
                    # Esquina inferior izquierda
                    cv2.line(frame, (box[0], box[3]), (box[0] + corner_length, box[3]), color_box, corner_thick)
                    cv2.line(frame, (box[0], box[3]), (box[0], box[3] - corner_length), color_box, corner_thick)
                    # Esquina inferior derecha
                    cv2.line(frame, (box[2], box[3]), (box[2] - corner_length, box[3]), color_box, corner_thick)
                    cv2.line(frame, (box[2], box[3]), (box[2], box[3] - corner_length), color_box, corner_thick)

                    # Etiqueta "SOSPECHOSO DETECTADO" encima del box
                    label = "SOSPECHOSO DETECTADO"
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    label_y = max(box[1] - 10, label_size[1] + 10)

                    # Fondo semi-transparente para la etiqueta
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (box[0], label_y - label_size[1] - 10),
                                 (box[0] + label_size[0] + 10, label_y + 5), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                    # Texto de la etiqueta
                    cv2.putText(frame, label, (box[0] + 5, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Confianza debajo del box
                    conf_text = f"Conf: {conf:.0%}"
                    cv2.putText(frame, conf_text, (box[0], box[3] + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Centro del objetivo
                    cx = (box[0] + box[2]) // 2
                    cy = (box[1] + box[3]) // 2
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2)

                    # Línea de tracking (desde centro a objetivo)
                    cv2.line(frame, (w//2, h//2), (cx, cy), (0, 255, 255), 2)

                # Retícula central moderna
                crosshair_size = 40
                crosshair_color = (0, 255, 0)
                # Líneas horizontales
                cv2.line(frame, (w//2 - crosshair_size, h//2), (w//2 - 10, h//2), crosshair_color, 2)
                cv2.line(frame, (w//2 + 10, h//2), (w//2 + crosshair_size, h//2), crosshair_color, 2)
                # Líneas verticales
                cv2.line(frame, (w//2, h//2 - crosshair_size), (w//2, h//2 - 10), crosshair_color, 2)
                cv2.line(frame, (w//2, h//2 + 10), (w//2, h//2 + crosshair_size), crosshair_color, 2)
                # Círculo central
                cv2.circle(frame, (w//2, h//2), 8, crosshair_color, 2)
                cv2.circle(frame, (w//2, h//2), 3, crosshair_color, -1)

                # Zona muerta (verde)
                zw = int(w * ZONA_MUERTA)
                zh = int(h * ZONA_MUERTA)
                cv2.rectangle(frame, (w//2 - zw, h//2 - zh), (w//2 + zw, h//2 + zh), (0, 255, 0), 1)

                # Zona precisión (amarillo)
                pw = int(w * ZONA_PRECISION)
                ph = int(h * ZONA_PRECISION)
                cv2.rectangle(frame, (w//2 - pw, h//2 - ph), (w//2 + pw, h//2 + ph), (0, 255, 255), 1)

                # Panel de información moderno (esquina superior izquierda)
                fps = info.get('fps', 0)
                conf = info.get('conf', 0)
                estado = info.get('estado', 'BUSCANDO')
                error = info.get('error', (0, 0))
                vel = info.get('velocidad', (0, 0))

                # Fondo semi-transparente para el panel
                panel_h = 180
                panel_w = 340
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                # Borde del panel
                cv2.rectangle(frame, (10, 10), (panel_w, panel_h), (0, 255, 255), 2)

                # Título del panel
                cv2.putText(frame, "SISTEMA DE VIGILANCIA", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.line(frame, (20, 42), (panel_w - 10, 42), (0, 255, 255), 1)

                # Información
                y_offset = 65
                line_height = 28

                cv2.putText(frame, f"FPS: {fps}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if conf > 0:
                    cv2.putText(frame, f"Confianza: {conf:.0%}", (20, y_offset + line_height),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, f"Confianza: --", (20, y_offset + line_height),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.putText(frame, f"Error: ({error[0]:+.2f}, {error[1]:+.2f})",
                           (20, y_offset + line_height * 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.putText(frame, f"Velocidad: ({vel[0]:+.2f}, {vel[1]:+.2f})",
                           (20, y_offset + line_height * 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Estado con colores dinámicos
                if estado == "TRACKING":
                    color_estado = (0, 0, 255)  # Rojo para tracking activo
                    estado_text = "RASTREANDO"
                elif estado == "CENTRADO":
                    color_estado = (0, 255, 0)  # Verde para centrado
                    estado_text = "CENTRADO"
                else:
                    color_estado = (0, 165, 255)  # Naranja para buscando
                    estado_text = "BUSCANDO..."

                cv2.putText(frame, f"Estado: {estado_text}", (20, y_offset + line_height * 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_estado, 2)

                # Timestamp (esquina inferior derecha)
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (w - 120, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow("SISTEMA DE VIGILANCIA PTZ", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
    
    def is_running(self):
        return self.running
    
    def stop(self):
        self.running = False


class TrackerPTZ:
    """Sistema principal con control PID y base de datos"""
    def __init__(self):
        self.running = True
        self.stats = {'frames': 0, 'detections': 0, 'moves': 0}

        print("\n" + "="*50)
        print("  SISTEMA DE SEGUIMIENTO PTZ - ESTABLE")
        print("="*50)

        print("\n[1/6] Conectando a base de datos...")
        self.db = ControladorDB()

        print("[2/6] Conectando a cámara...")
        self.ptz = PTZController(IP, ONVIF_PORT, ONVIF_USER, ONVIF_PASS)

        print("[3/6] Iniciando stream...")
        self.stream = StreamCapture(RTSP_URL, BUFFER_SIZE)
        time.sleep(2)

        print("[4/6] Cargando detector...")
        self.detector = Detector(MODELO_YOLO, USAR_PERSONA_GENERAL)

        print("[5/6] Inicializando controladores PID...")
        self.pid_pan = ControladorPID(KP, KI, KD)
        self.pid_tilt = ControladorPID(KP, KI, KD)
        self.filtro = FiltroKalmanSimple(HISTORIAL_SIZE)

        # Inicializar grabador de video
        self.video_grabador = VideoGrabador()
        self.detecciones_consecutivas = 0  # Contador para evitar falsos positivos

        self.visualizador = None
        if MOSTRAR_VIDEO:
            print("[6/6] Iniciando visualización...")
            self.visualizador = Visualizador()
        else:
            print("[6/6] Modo headless")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n[OK] Sistema listo")
        print(f"     PID: Kp={KP}, Ki={KI}, Kd={KD}")
        print(f"     Zona muerta: {ZONA_MUERTA*100:.0f}%")
        print(f"     Control DB: Activo (chequeo cada {DB_CHECK_INTERVAL}s)")
        if EVOLUTION_BASE_URL and NUMERO_WHATSAPP:
            print(f"     WhatsApp: Activo (envío a {NUMERO_WHATSAPP})")
        else:
            print(f"     WhatsApp: Desactivado (configurar .env)")
        print("     Ctrl+C o 'Q' para detener")
        print("-"*50)
    
    def _signal_handler(self, sig, frame):
        print("\n\n[!] Deteniendo sistema...")
        self.running = False
    
    def run(self):
        frame_idx = 0
        frames_sin_deteccion = 0
        vel_pan = 0
        vel_tilt = 0
        ultimo_chequeo_db = time.time()

        while self.running:
            # Verificar estado de encendido en la base de datos
            if time.time() - ultimo_chequeo_db >= DB_CHECK_INTERVAL:
                if not self.db.verificar_encendido():
                    print("\n[DB] Sistema apagado desde base de datos")
                    self.running = False
                    break
                ultimo_chequeo_db = time.time()

            if self.visualizador and not self.visualizador.is_running():
                self.running = False
                break

            ts, frame = self.stream.get_latest()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_idx += 1
            self.stats['frames'] += 1
            
            if frame_idx % SKIP_FRAMES != 0:
                continue
            
            h, w = frame.shape[:2]
            
            info = {
                'fps': self.stream.get_fps(),
                'conf': 0,
                'estado': 'BUSCANDO',
                'error': (0, 0),
                'velocidad': (vel_pan, vel_tilt),
                'bbox': None
            }
            
            deteccion = self.detector.detectar(frame)
            
            if deteccion:
                px, py, conf, bbox = deteccion
                self.stats['detections'] += 1
                frames_sin_deteccion = 0
                self.detecciones_consecutivas += 1

                # Normalizar posición (0-1)
                px_norm = px / w
                py_norm = py / h

                # Filtrar posición para suavizar
                px_filtrado, py_filtrado = self.filtro.actualizar(px_norm, py_norm)

                # Error desde centro (0.5)
                ex = px_filtrado - 0.5
                ey = py_filtrado - 0.5

                info['conf'] = conf
                info['error'] = (ex, ey)
                info['bbox'] = bbox

                # Grabar y enviar video si hay detección estable
                if (self.detecciones_consecutivas >= 10 and  # Al menos 10 frames consecutivos
                    conf >= 0.6 and  # Confianza alta
                    self.video_grabador.puede_enviar() and  # Ha pasado el cooldown
                    not self.video_grabador.grabando):  # No está grabando actualmente

                    print(f"\n[ALERTA] Sospechoso detectado con confianza {conf:.2%}")
                    print("[ALERTA] Iniciando grabación y envío por WhatsApp...")

                    # Ejecutar en thread separado para no bloquear el tracking
                    def grabar_y_enviar():
                        video_path = self.video_grabador.grabar_video(self.stream)
                        if video_path:
                            self.video_grabador.enviar_video_whatsapp(video_path)

                    thread_video = threading.Thread(target=grabar_y_enviar, daemon=True)
                    thread_video.start()
                
                # Determinar zona y acción
                error_mag = np.sqrt(ex**2 + ey**2)
                
                if error_mag < ZONA_MUERTA:
                    # Dentro de zona muerta - parar
                    self.ptz.parar()
                    self.pid_pan.reset()
                    self.pid_tilt.reset()
                    vel_pan = 0
                    vel_tilt = 0
                    info['estado'] = 'CENTRADO'
                else:
                    # Calcular velocidad con PID
                    vel_pan = self.pid_pan.calcular(ex)
                    vel_tilt = self.pid_tilt.calcular(ey)
                    
                    # Aplicar inversiones
                    if INVERTIR_PAN:
                        vel_pan = -vel_pan
                    if INVERTIR_TILT:
                        vel_tilt = -vel_tilt
                    
                    # Reducir velocidad en zona de precisión
                    if error_mag < ZONA_PRECISION:
                        factor = error_mag / ZONA_PRECISION
                        vel_pan *= factor
                        vel_tilt *= factor
                    
                    self.ptz.mover(pan=vel_pan, tilt=vel_tilt)
                    self.stats['moves'] += 1
                    info['estado'] = 'TRACKING'
                
                info['velocidad'] = (vel_pan, vel_tilt)
                
                # Log
                if self.stats['frames'] % 100 == 0:
                    fps = self.stream.get_fps()
                    print(f"[TRACK] FPS:{fps} | Conf:{conf:.2f} | "
                          f"Err:({ex:+.2f},{ey:+.2f}) | "
                          f"Vel:({vel_pan:+.3f},{vel_tilt:+.3f})")
            else:
                frames_sin_deteccion += 1
                self.detecciones_consecutivas = 0  # Resetear contador

                if frames_sin_deteccion > 20:
                    self.ptz.parar()
                    self.pid_pan.reset()
                    self.pid_tilt.reset()
                    self.filtro.reset()
                    vel_pan = 0
                    vel_tilt = 0

                    if frames_sin_deteccion == 21:
                        print("[WARN] Objetivo perdido")
            
            if self.visualizador:
                self.visualizador.actualizar(frame, info)
        
        self.cleanup()
    
    def cleanup(self):
        print("\n[LIMPIEZA]")
        self.ptz.parar()
        self.stream.release()
        if self.visualizador:
            self.visualizador.stop()

        # Actualizar base de datos
        self.db.apagar_sistema()
        self.db.cerrar()

        print(f"\n[STATS] Frames:{self.stats['frames']} | "
              f"Detecciones:{self.stats['detections']} | "
              f"Movimientos:{self.stats['moves']}")
        print("[OK] Sistema detenido")


def main():
    print("\n" + "="*60)
    print("  SISTEMA DE SEGUIMIENTO PTZ CON CONTROL DB")
    print("="*60)
    print("\n[INFO] Monitoreando base de datos...")
    print(f"[INFO] Esperando que encendido = TRUE en control_dron (id=1)")
    print("[INFO] Presiona Ctrl+C para cancelar")

    # Crear controlador de DB para monitoreo inicial
    db_monitor = ControladorDB()

    try:
        # Esperar hasta que encendido = TRUE
        while True:
            if db_monitor.verificar_encendido():
                print("\n[DB] ¡Sistema ENCENDIDO detectado!")
                print("[INFO] Iniciando sistema de seguimiento...\n")
                db_monitor.cerrar()
                break
            else:
                print(f"[DB] Esperando... (encendido = FALSE) - {time.strftime('%H:%M:%S')}")
                time.sleep(DB_CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n[INFO] Monitoreo cancelado por usuario")
        db_monitor.cerrar()
        sys.exit(0)

    # Iniciar sistema de tracking
    try:
        tracker = TrackerPTZ()
        tracker.run()
    except Exception as e:
        print(f"\n[ERROR] Error en sistema de tracking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()