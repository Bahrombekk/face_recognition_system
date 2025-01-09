import cv2
import os
import time
import numpy as np
from mtcnn import MTCNN
import face_recognition
from config import Config

class DataCollector:
    def __init__(self):
        self.config = Config()
        self.detector = MTCNN()
        
    def collect_user_data(self, user_id: str):
        """Foydalanuvchi ma'lumotlarini yig'ish"""
        user_dir = os.path.join(self.config.DATASET_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        collected_positions = set()
        
        print(f"\nFoydalanuvchi {user_id} uchun rasmlar yig'ish boshlandi")
        
        for position in self.config.REQUIRED_FACE_POSITIONS:
            if position in collected_positions:
                continue
                
            print(f"\n{position} holatida kameraga qarang...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Yuzni aniqlash
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb_frame)
                
                if faces:
                    face = faces[0]
                    x, y, w, h = face['box']
                    
                    # Yuz sohasini ajratib olish
                    face_area = frame[
                        max(0, y-30):min(frame.shape[0], y+h+30),
                        max(0, x-30):min(frame.shape[1], x+w+30)
                    ]
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow('Face Collection', frame)
                key = cv2.waitKey(1)
                
                if key == 32:  # Space bosilganda
                    if self._check_face_quality(face_area):
                        img_path = os.path.join(user_dir, f"{position}.jpg")
                        cv2.imwrite(img_path, face_area)
                        collected_positions.add(position)
                        print(f"{position} holati saqlandi")
                        break
                    else:
                        print("Yuz sifati yaxshi emas, qayta urinib ko'ring")
                
                elif key == 27:  # ESC bosilganda
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return len(collected_positions) == len(self.config.REQUIRED_FACE_POSITIONS)
    
    def _check_face_quality(self, face_image) -> bool:
        """Yuz sifatini tekshirish"""
        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            return len(encodings) > 0
        except:
            return False
    def delete_user_data(self, user_id: str):
        """Foydalanuvchi ma'lumotlarini o'chirish"""
        user_dir = os.path.join(self.config.DATASET_DIR, str(user_id))
        if os.path.exists(user_dir):
            os.rmdir(user_dir)
            print(f"Foydalanuvchi {user_id} ma'lumotlari o'chirildi")
        else:
            print(f"Foydalanuvchi {user_id} ma'lumotlari mavjud emas")
