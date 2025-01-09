from sklearn.cluster import KMeans
import numpy as np
from typing import Dict, List, Tuple

class FaceClusterManager:
    def __init__(self, n_clusters: int = 100):
        self.max_clusters = n_clusters
        self.kmeans = None
        self.user_clusters: Dict[int, Dict[str, np.ndarray]] = {}
        self.cluster_centers: Dict[int, np.ndarray] = {}
        
    def create_clusters(self, encodings_db: Dict[str, np.ndarray]) -> None:
        """Foydalanuvchilarni clusterlarga bo'lish"""
        if not encodings_db:
            print("No encodings found in database")
            return
            
        encodings_list = list(encodings_db.values())
        user_ids = list(encodings_db.keys())
        
        # Determine appropriate number of clusters based on data size
        n_samples = len(encodings_list)
        n_clusters = min(self.max_clusters, max(1, n_samples // 2))
        
        if n_samples < 2:
            # Handle case with very few samples
            self.cluster_centers = {0: np.mean(encodings_list, axis=0)}
            self.user_clusters = {0: encodings_db}
            return
            
        # Initialize KMeans with appropriate number of clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(encodings_list)
        
        # Store cluster centers
        self.cluster_centers = {
            i: center for i, center in enumerate(self.kmeans.cluster_centers_)
        }
        
        # Group users by cluster
        self.user_clusters = {}
        for user_id, cluster_id in zip(user_ids, cluster_labels):
            if cluster_id not in self.user_clusters:
                self.user_clusters[cluster_id] = {}
            self.user_clusters[cluster_id][user_id] = encodings_db[user_id]
            
    def find_closest_clusters(self, encoding: np.ndarray, n: int = 3) -> List[Tuple[float, int]]:
        """Eng yaqin clusterlarni topish"""
        if not self.cluster_centers:
            return [(0.0, 0)]  # Return single cluster if no clustering performed
            
        distances = []
        for cluster_id, center in self.cluster_centers.items():
            dist = np.linalg.norm(encoding - center)
            distances.append((dist, cluster_id))
        
        n = min(n, len(distances))  # Ensure we don't try to return more clusters than exist
        return sorted(distances)[:n]
    def all_users_in_cluster(self, cluster_id: int) -> List[str]:
        """Clusterdagi barcha foydalanuvchilar"""
        return list(self.user_clusters[cluster_id].keys())
    
    #Encoding baced on the data in the datasets,updating the database, i.e. adding missing data
    def update_database(self, new_encodings: Dict[str, np.ndarray]) -> List[str]:
        """
        Database asosida yangi encodinglarni qo'shish.
        Mavjud foydalanuvchilarning ma'lumotlariga tegilmaydi.

        :param new_encodings: Yangi encodinglarni o'z ichiga olgan dictionary.
                          Format: {user_id: encodings}
        :return: Yangi qo'shilgan foydalanuvchilar ro'yxati.
        """
        added_users = []
        for user_id,encodings in new_encodings.items():
            # Tekshirish : ID bazada  mavjudligini tekshirish
            if self.has_encoding(user_id):
                print(f"{user_id} allaqachon mavjud.")
            else:
                print(f"{user_id} uchun encoding qo'shilmoqda...")
                self.add_encoding(user_id, encodings)
                added_users.append(user_id)
        return added_users
                 

