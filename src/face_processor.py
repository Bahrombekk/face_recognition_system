import face_recognition
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

class FaceProcessor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        
    def detect_faces(self, frame: np.ndarray) -> Tuple[List, List]:
        """Kadrdagi yuzlarni aniqlash"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings
    
    def process_faces_parallel(self, frame: np.ndarray, recognition_func) -> List:
        """Yuzlarni parallel ravishda qayta ishlash"""
        face_locations, face_encodings = self.detect_faces(frame)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(recognition_func, face_encodings))
            
        return results, face_locations