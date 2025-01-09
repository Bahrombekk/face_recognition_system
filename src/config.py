import os

class Config:
    # Asosiy manzillar
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Ma'lumotlar bazasi manzillari
    DATABASE_DIR = os.path.join(BASE_DIR, "database")
    LMDB_PATH = os.path.join(DATABASE_DIR, "face_encodings.lmdb")
    
    # Dataset manzillari
    DATASET_DIR = os.path.join(BASE_DIR, "/home/bahrombek/Desktop/faceid/dataset")
    
    # Log manzili
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # Modellar sozlamalari
    FACE_DETECTION_MODEL = "hog"  # yoki 'cnn' GPU bo'lsa
    CLUSTER_COUNT = 100
    ENCODING_TOLERANCE = 0.6
    MIN_CONFIDENCE = 0.7
    
    # Face recognition parametrlari
    REQUIRED_FACE_POSITIONS = [
        "straight",
        "left",
        "right",
        "smile",
        "light_dark"
    ]
    
    def __init__(self):
        # Kerakli papkalarni yaratish
        self._create_directories()
    
    def _create_directories(self):
        """Kerakli papkalarni yaratish"""
        directories = [
            self.DATABASE_DIR,
            self.DATASET_DIR,
            self.LOG_DIR
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    def _dalete_directories(self):
        """Kerakli papkalarni o'chirish"""
        directories = [
            self.DATABASE_DIR,
            self.DATASET_DIR,
            self.LOG_DIR
        ]
        
        for directory in directories:
            if os.path.exists(directory):
                os.rmdir(directory)
    def _all_users_classes_data(self):
        return os.listdir(self.DATASET_DIR)
    
    