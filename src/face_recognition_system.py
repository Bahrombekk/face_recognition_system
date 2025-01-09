# face_recognition_system.py
import numpy as np
from functools import lru_cache
from typing import Tuple, Optional, List
import cv2
from database import EncodingDatabase
from clustering import FaceClusterManager
from face_processor import FaceProcessor
from config import Config

class FaceRecognitionSystem:
    def __init__(self):
        self.config = Config()
        self.db = EncodingDatabase()
        self.cluster_manager = FaceClusterManager(self.config.CLUSTER_COUNT)
        self.face_processor = FaceProcessor()
        self.initialize_system()
        
    def initialize_system(self) -> None:
        """Tizimni ishga tushirish"""
        try:
            encodings = self.db.get_all_encodings()
            if not encodings:
                print("Warning: No face encodings found in database")
            self.cluster_manager.create_clusters(encodings)
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """Kadrni qayta ishlash"""
        # Get face locations and encodings
        face_locations, face_encodings = self.face_processor.detect_faces(frame)
        results = []
        
        # Process each detected face
        for face_encoding in face_encodings:
            # Try to recognize the face
            user_id, confidence = self.recognize_face(face_encoding)
            results.append((user_id, confidence))
            
        # Draw results on frame
        for (top, right, bottom, left), (user_id, confidence) in zip(face_locations, results):
            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add text with user ID and confidence
            text = "Unknown"
            if user_id and confidence:
                if confidence > self.config.MIN_CONFIDENCE:
                    text = f"{user_id} ({confidence:.1%})"
            
            # Calculate text position and add background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, 
                         (left, top - text_size[1] - 10), 
                         (left + text_size[0], top), 
                         (0, 255, 0), 
                         cv2.FILLED)
            
            # Add text
            cv2.putText(frame, text, 
                       (left, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
        
        return frame, results
    
    def run_recognition(self):
        """Real-time yuz tanib olishni ishga tushirish"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
                
            print("\nYuz tanib olish tizimi ishga tushdi")
            print("Chiqish uchun 'q' tugmasini bosing")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                    
                # Process the frame
                processed_frame, results = self.process_frame(frame)
                
                # Display the result
                cv2.imshow('Face Recognition', processed_frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during recognition: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
    
    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Yuzni tanib olish"""
        closest_clusters = self.cluster_manager.find_closest_clusters(face_encoding)
        
        best_match = None
        min_distance = float('inf')
        
        for _, cluster_id in closest_clusters:
            if cluster_id in self.cluster_manager.user_clusters:
                for user_id, user_encoding in self.cluster_manager.user_clusters[cluster_id].items():
                    distance = np.linalg.norm(face_encoding - user_encoding)
                    if distance < min_distance and distance < self.config.ENCODING_TOLERANCE:
                        min_distance = distance
                        best_match = user_id
                    
        return best_match, 1 - min_distance if best_match else None