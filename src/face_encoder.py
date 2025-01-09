import face_recognition
import cv2
import os
import numpy as np
from typing import Dict, Optional
from config import Config
from database import EncodingDatabase
class FaceEncoder:
    def __init__(self):
        self.config = Config()
        self.db = EncodingDatabase()
    
    def encode_user(self, user_id: str) -> Optional[np.ndarray]:
        """Foydalanuvchi uchun encoding yaratish"""
        user_dir = os.path.join(self.config.DATASET_DIR, str(user_id))
        if not os.path.exists(user_dir):
            print(f"Xato: {user_id} uchun ma'lumotlar topilmadi")
            return None
            
        encodings = []
        
        for position in self.config.REQUIRED_FACE_POSITIONS:
            img_path = os.path.join(user_dir, f"{position}.jpg")
            if not os.path.exists(img_path):
                print(f"Xato: {position} holati uchun rasm topilmadi")
                continue
                
            encoding = self._encode_image(img_path)
            if encoding is not None:
                encodings.append(encoding)
        
        if not encodings:
            return None
            
        # O'rtacha encoding hisoblash
        mean_encoding = np.mean(encodings, axis=0)
        
        # Ma'lumotlar bazasiga saqlash
        self.db.save_encoding(str(user_id), mean_encoding)
        
        return mean_encoding
    
    def _encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """Rasmdan encoding olish"""
        try:
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            return encodings[0] if encodings else None
        except Exception as e:
            print(f"Xatolik: {image_path} ni encode qilishda xatolik: {str(e)}")
            return None
    def _dalete_encoding(self, user_id: str) -> None:
        """Foydalanuvchi encodingini o'chirish"""
        self.db.delete_encoding(str(user_id))

    #all user get_all_user_ids
    def get_all_user_ids(self) -> Dict[str, np.ndarray]:
        return self.db.get_all_user_ids()
    def get_all_encodings(self) -> Dict[str, np.ndarray]:
        return self.db.get_all_encodings()