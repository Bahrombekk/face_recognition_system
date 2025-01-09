import lmdb
import pickle
import numpy as np
from typing import Dict, Optional
from config import Config

class EncodingDatabase:
    def __init__(self):
        self.config = Config()
        self.env = lmdb.open(
            self.config.LMDB_PATH, 
            map_size=10*(1024*1024*1024)
        )
        
    def save_encoding(self, user_id: str, encoding: np.ndarray) -> None:
        """Foydalanuvchi encodingini saqlash"""
        with self.env.begin(write=True) as txn:
            txn.put(
                user_id.encode(), 
                pickle.dumps(encoding)
            )
            
    def get_encoding(self, user_id: str) -> Optional[np.ndarray]:
        """Foydalanuvchi encodingini olish"""
        with self.env.begin() as txn:
            encoding_bytes = txn.get(user_id.encode())
            return pickle.loads(encoding_bytes) if encoding_bytes else None
    
    def get_all_encodings(self) -> Dict[str, np.ndarray]:
        """Barcha encodinglarni olish"""
        encodings = {}
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                user_id = key.decode()
                encoding = pickle.loads(value)
                encodings[user_id] = encoding
        return encodings

    def delete_encoding(self, user_id: str) -> None:
        """Foydalanuvchi encodingini o'chirish"""
        with self.env.begin(write=True) as txn:
            txn.delete(user_id.encode())

    def get_all_user_ids(self) -> Dict[str, np.ndarray]:
        """Barcha foydalanuvchilarni olish"""
        user_ids = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                user_ids.append(key.decode())
        return user_ids

