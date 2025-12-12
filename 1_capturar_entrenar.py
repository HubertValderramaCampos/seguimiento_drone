import os
import sys
import cv2
import time
import shutil
import threading
import numpy as np
from pathlib import Path
from collections import deque

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# ============== CONFIGURACIÓN ==============
IP = "192.168.101.19"
RTSP_URL = f"rtsp://{IP}:554/user=admin&password=&channel=1&stream=0.sdp"

PERSONA_NOMBRE = "objetivo"
DATASET_DIR = Path("dataset_persona")
NUM_FRAMES_OBJETIVO = 200  # Frames a capturar
MIN_CONFIANZA = 0.6  # Confianza mínima para capturar
MIN_AREA_RATIO = 0.02  # Área mínima de bbox (2% del frame)
MAX_AREA_RATIO = 0.7  # Área máxima (70% del frame)
INTERVALO_CAPTURA = 0.1  # Segundos entre capturas (más rápido con GPU)
VARIACION_MIN = 25  # Píxeles de movimiento para considerar nueva pose

# Entrenamiento (Optimizado para GPU)
EPOCHS = 50
BATCH_SIZE = 16  # Aumentado para GPU
IMG_SIZE = 640
WORKERS = 8  # Threads paralelos para carga de datos
# ==========================================


def detectar_device():
    """Detecta si hay GPU disponible"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU] {gpu_name} ({gpu_memory:.1f} GB VRAM)")
            print(f"[GPU] CUDA {torch.version.cuda}")
            return 0
    except Exception as e:
        print(f"[!] Error detectando GPU: {e}")
    print("[CPU] Usando CPU (MUY LENTO)")
    return 'cpu'


class StreamBuffer:
    """Buffer de stream sin pérdida"""
    def __init__(self, url, buffer_size=30):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.buffer = deque(maxlen=buffer_size)
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        time.sleep(1)
    
    def _reader(self):
        while self.running:
            try:
                if not self.cap.isOpened():
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    continue
                    
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.buffer.append(frame)
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.5)
    
    def read(self):
        with self.lock:
            return self.buffer[-1].copy() if self.buffer else None
    
    def release(self):
        self.running = False
        time.sleep(0.2)
        try:
            self.cap.release()
        except:
            pass


class DetectorPersona:
    """Detector de personas con YOLOv8 acelerado por GPU"""
    def __init__(self):
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            print("[!] Instalando ultralytics...")
            os.system("pip install ultralytics -q")
            from ultralytics import YOLO
            self.YOLO = YOLO

        print("[+] Cargando YOLOv8 en GPU...")
        # Forzar uso de GPU con device=0
        self.model = self.YOLO("yolov8n.pt")
        self.device = detectar_device()

        # Warmup GPU
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device)
        print(f"[OK] Detector listo en {'GPU' if self.device == 0 else 'CPU'}")
    
    def detectar_persona(self, frame):
        """
        Detecta persona más grande/cercana con GPU
        Retorna: dict con info o None
        """
        h, w = frame.shape[:2]
        # Ejecutar en GPU con half precision para mayor velocidad
        results = self.model(frame, verbose=False, classes=[0], device=self.device, half=True if self.device == 0 else False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Calcular áreas
            xyxy = boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            confs = boxes.conf.cpu().numpy()
            
            # Filtrar por confianza
            valid = confs >= MIN_CONFIANZA
            if not valid.any():
                return None
            
            # Tomar la más grande entre las válidas
            areas_valid = np.where(valid, areas, 0)
            idx = areas_valid.argmax()
            
            bbox = xyxy[idx]
            conf = confs[idx]
            area_ratio = areas[idx] / (w * h)
            
            # Verificar tamaño
            if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                return None
            
            # Formato YOLO normalizado
            x_center = ((bbox[0] + bbox[2]) / 2) / w
            y_center = ((bbox[1] + bbox[3]) / 2) / h
            box_w = (bbox[2] - bbox[0]) / w
            box_h = (bbox[3] - bbox[1]) / h
            
            return {
                'xyxy': bbox,
                'xywh_norm': (x_center, y_center, box_w, box_h),
                'conf': conf,
                'area_ratio': area_ratio,
                'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            }
        return None


class CapturadorInteligente:
    """Captura automática con variedad de poses"""
    def __init__(self):
        self.detector = DetectorPersona()
        self.stream = StreamBuffer(RTSP_URL)
        self.frames_capturados = []
        self.ultima_captura = 0
        self.ultimo_centro = None
        
        # Preparar dataset
        self._preparar_dataset()
    
    def _preparar_dataset(self):
        """Crea estructura YOLO"""
        for split in ['train', 'val']:
            (DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        yaml_content = f"""path: {DATASET_DIR.absolute()}
train: train/images
val: val/images

names:
  0: {PERSONA_NOMBRE}
"""
        with open(DATASET_DIR / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
    
    def _es_pose_nueva(self, centro):
        """Verifica si la pose es suficientemente diferente"""
        if self.ultimo_centro is None:
            return True
        
        dx = abs(centro[0] - self.ultimo_centro[0])
        dy = abs(centro[1] - self.ultimo_centro[1])
        
        return (dx > VARIACION_MIN or dy > VARIACION_MIN)
    
    def _guardar_frame(self, frame, deteccion, idx):
        """Guarda frame y etiqueta"""
        split = 'train' if idx % 5 != 0 else 'val'
        
        img_path = DATASET_DIR / split / 'images' / f"{PERSONA_NOMBRE}_{idx:04d}.jpg"
        label_path = DATASET_DIR / split / 'labels' / f"{PERSONA_NOMBRE}_{idx:04d}.txt"
        
        # Guardar imagen
        cv2.imwrite(str(img_path), frame)
        
        # Guardar etiqueta YOLO
        x, y, w, h = deteccion['xywh_norm']
        with open(label_path, 'w') as f:
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        return True
    
    def capturar(self):
        """Loop principal de captura automática"""
        print("\n" + "="*60)
        print("  CAPTURA AUTOMÁTICA DE PERSONA")
        print("="*60)
        print(f"\nObjetivo: {NUM_FRAMES_OBJETIVO} frames")
        print("La persona debe moverse para capturar diferentes poses")
        print("Presiona 'Q' para terminar antes")
        print("-"*60 + "\n")
        
        cv2.namedWindow("Captura Auto", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Captura Auto", 960, 540)
        
        frames_guardados = 0
        persona_detectada = False
        tiempo_sin_persona = 0
        
        while frames_guardados < NUM_FRAMES_OBJETIVO:
            frame = self.stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            display = frame.copy()
            h, w = frame.shape[:2]
            
            # Detectar persona
            deteccion = self.detector.detectar_persona(frame)
            
            if deteccion:
                persona_detectada = True
                tiempo_sin_persona = 0
                
                bbox = deteccion['xyxy'].astype(int)
                conf = deteccion['conf']
                centro = deteccion['center']
                
                # Dibujar bbox
                color = (0, 255, 0)
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(display, f"Conf: {conf:.2f}", (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Verificar si capturar
                ahora = time.time()
                puede_capturar = (ahora - self.ultima_captura) >= INTERVALO_CAPTURA
                es_nueva = self._es_pose_nueva(centro)
                
                if puede_capturar and es_nueva:
                    if self._guardar_frame(frame, deteccion, frames_guardados):
                        frames_guardados += 1
                        self.ultima_captura = ahora
                        self.ultimo_centro = centro
                        
                        cv2.rectangle(display, (0, 0), (w, 10), (0, 255, 0), -1)
                        print(f"[+] Frame {frames_guardados}/{NUM_FRAMES_OBJETIVO} | "
                              f"Conf: {conf:.2f} | Pos: ({int(centro[0])}, {int(centro[1])})")
                
                if not es_nueva and puede_capturar:
                    cv2.putText(display, "MUEVETE!", (w//2 - 80, h - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            else:
                tiempo_sin_persona += 1
                if tiempo_sin_persona > 30:
                    cv2.putText(display, "BUSCANDO PERSONA...", (w//2 - 150, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Barra de progreso
            progreso = frames_guardados / NUM_FRAMES_OBJETIVO
            bar_w = int(w * 0.8)
            bar_x = int(w * 0.1)
            cv2.rectangle(display, (bar_x, 20), (bar_x + bar_w, 50), (50, 50, 50), -1)
            cv2.rectangle(display, (bar_x, 20), (bar_x + int(bar_w * progreso), 50), (0, 255, 0), -1)
            cv2.putText(display, f"{frames_guardados}/{NUM_FRAMES_OBJETIVO}", 
                       (bar_x + bar_w//2 - 40, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            estado = "CAPTURANDO" if persona_detectada else "ESPERANDO"
            cv2.putText(display, estado, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if persona_detectada else (0, 0, 255), 2)
            
            cv2.imshow("Captura Auto", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[!] Captura interrumpida por usuario")
                break
        
        cv2.destroyAllWindows()
        self.stream.release()
        
        print(f"\n[OK] Captura completada: {frames_guardados} frames")
        print(f"    Train: {len(list((DATASET_DIR / 'train' / 'images').glob('*.jpg')))} imgs")
        print(f"    Val: {len(list((DATASET_DIR / 'val' / 'images').glob('*.jpg')))} imgs")
        
        return frames_guardados


def entrenar_modelo():
    """Entrena YOLOv8 con el dataset capturado"""
    print("\n" + "="*60)
    print("  ENTRENAMIENTO DE MODELO PERSONALIZADO")
    print("="*60 + "\n")
    
    from ultralytics import YOLO
    
    # Detectar device
    device = detectar_device()
    
    # Verificar dataset
    train_imgs = len(list((DATASET_DIR / 'train' / 'images').glob('*.jpg')))
    val_imgs = len(list((DATASET_DIR / 'val' / 'images').glob('*.jpg')))
    
    if train_imgs < 10:
        print("[ERROR] Dataset muy pequeño, necesitas al menos 10 imágenes")
        return None
    
    print(f"Dataset: {train_imgs} train, {val_imgs} val")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    
    if device == 'cpu':
        print("\n[!] ADVERTENCIA: Entrenar en CPU es LENTO (1-3 horas)")
        print("    Para GPU instala: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        resp = input("\n¿Continuar en CPU? (s/n): ").lower()
        if resp != 's':
            return None
    
    # Cargar modelo base
    model = YOLO("yolov8n.pt")

    # Configuración optimizada según device
    if device == 'cpu':
        workers = 0
        batch = 4
        cache = False
        print("\n[!] Entrenamiento en CPU será LENTO")
    else:
        workers = WORKERS
        batch = BATCH_SIZE
        cache = True  # Cache en RAM para GPU
        print(f"\n[GPU] Configuración: batch={batch}, workers={workers}")

    # Entrenar con configuración GPU optimizada
    results = model.train(
        data=str(DATASET_DIR / 'dataset.yaml'),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch,
        name=f'{PERSONA_NOMBRE}_detector',
        patience=20,
        save=True,
        device=device,
        workers=workers,
        cache=cache,  # Cache en RAM
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,  # Copy-paste augmentation
        degrees=15,
        translate=0.15,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
        amp=True if device == 0 else False,  # Automatic Mixed Precision en GPU
        plots=True,
        close_mosaic=10  # Últimas epochs sin mosaic para mejor precisión
    )
    
    # Copiar mejor modelo
    best_path = Path(f'runs/detect/{PERSONA_NOMBRE}_detector/weights/best.pt')
    if best_path.exists():
        output = f'{PERSONA_NOMBRE}_modelo.pt'
        shutil.copy(best_path, output)
        print(f"\n[OK] Modelo guardado: {output}")
        return output
    
    # Buscar alternativas
    for p in Path('runs/detect').rglob('best.pt'):
        output = f'{PERSONA_NOMBRE}_modelo.pt'
        shutil.copy(p, output)
        print(f"\n[OK] Modelo guardado: {output}")
        return output
    
    return None


def main():
    print("\n" + "="*60)
    print("  SISTEMA DE CAPTURA Y ENTRENAMIENTO AUTOMÁTICO")
    print("="*60)
    
    # Fase 1: Captura
    print("\n[FASE 1] Captura de frames")
    capturador = CapturadorInteligente()
    frames = capturador.capturar()
    
    if frames < 10:
        print("[!] Muy pocos frames capturados. Abortando.")
        return
    
    # Fase 2: Entrenamiento
    print("\n[FASE 2] Entrenamiento")
    respuesta = input("¿Iniciar entrenamiento? (s/n): ").lower()
    
    if respuesta == 's':
        modelo = entrenar_modelo()
        if modelo:
            print("\n" + "="*60)
            print(f"  ¡COMPLETADO!")
            print(f"  Modelo: {modelo}")
            print(f"  Úsalo en el script de seguimiento")
            print("="*60)
    else:
        print("\n[INFO] Dataset guardado. Puedes entrenar después.")


if __name__ == "__main__":
    main()