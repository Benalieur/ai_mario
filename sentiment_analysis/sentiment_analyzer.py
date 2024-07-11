import logging
import cv2
import torch
import threading
from facenet_pytorch import MTCNN
from deepface import DeepFace

class SentimentAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.emotions_list = []
        self.lock = threading.Lock()
        self.capture_interval = 30  # capture emotions every 30 frames
        self.frame_count = 0
        self.thread = None

    def get_webcam_frame(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return ret, frame
            else:
                self.logger.error("Failed to read frame from camera with index 0")
        else:
            self.logger.error("Failed to open camera with index 0")
        return False, None

    def capture_emotions(self):
        ret, frame = self.get_webcam_frame()
        if ret:
            _, emotions_list = self.analyze_frame(frame)
            if emotions_list:
                with self.lock:
                    self.emotions_list.extend(emotions_list)
        else:
            self.logger.error("Erreur : Impossible d'ouvrir la caméra.")

    def analyze_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(frame_rgb)
        emotions_list = []
        self.last_boxes = boxes
        if boxes is not None:
            for box in boxes:
                left, top, right, bottom = map(int, box)
                roi = frame_rgb[top:bottom, left:right]
                try:
                    analysis = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)[0]
                    emotions = analysis['emotion']
                    emotions_list.append(emotions)
                except Exception as e:
                    self.logger.error(f"Erreur dans la détection des émotions: {e}")
        return frame_rgb, emotions_list

    def capture_emotions_continuously(self):
        self.frame_count += 1
        if self.frame_count % self.capture_interval == 0:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.capture_emotions)
                self.thread.start()

    def get_dominant_emotion(self):
        if not self.emotions_list:
            return 'neutral'
        average_emotions = {}
        with self.lock:
            for emotions in self.emotions_list:
                for emotion, value in emotions.items():
                    if emotion in average_emotions:
                        average_emotions[emotion] += value
                    else:
                        average_emotions[emotion] = value
            for emotion in average_emotions:
                average_emotions[emotion] /= len(self.emotions_list)
            dominant_emotion = max(average_emotions, key=average_emotions.get)
            self.emotions_list = []  # Reset la liste après chaque utilisation
        return dominant_emotion

    def detect_persons(self, frame):
        if self.last_boxes is not None and len(self.last_boxes) > 0:
            return True
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.detect_bodies(frame_gray)
        return len(bodies) > 0

    def detect_bodies(self, frame_gray):
        body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        bodies = body_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return bodies
