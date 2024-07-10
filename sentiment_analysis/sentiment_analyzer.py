import logging
import cv2
import torch
from facenet_pytorch import MTCNN
from deepface import DeepFace


class SentimentAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialisation du détecteur d'émotions et de visages.
        """
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def analyze_frame(self, frame):
        """
        Analyse une image pour détecter les émotions.
        
        :param frame: Image à analyser
        :return: Image convertie en RGB, liste des émotions
        """
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

    def detect_persons(self, frame):
        """
        Détecte la présence de personnes dans l'image analysée précédemment.
        
        :param frame: Image à analyser
        :return: Booléen indiquant la présence de personnes
        """
        if self.last_boxes is not None and len(self.last_boxes) > 0:
            return True
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.detect_bodies(frame_gray)
        
        return len(bodies) > 0

    def detect_bodies(self, frame_gray):
        """
        Utilise le classificateur de Haar pour détecter des corps humains dans une image en niveaux de gris.
        
        :param frame_gray: Image en niveaux de gris à analyser
        :return: Liste de rectangles englobants pour les corps détectés
        """
        body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        bodies = body_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return bodies
