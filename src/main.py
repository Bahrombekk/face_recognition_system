from config import Config
from data_collector import DataCollector
from face_encoder import FaceEncoder
from face_recognition_system import FaceRecognitionSystem

def main():
    config = Config()
    
    while True:
        print("\nFace ID Tizimi")
        print("1. Yangi foydalanuvchi qo'shish")
        print("2. Yuzni tanib olishni boshlash")
        print("3. Chiqish")
        print("4  Foydalanuvchini o'chirish")
        print("5. Barcha foydalanuvchilarni ko'rish")
        
        choice = input("Tanlang (1-5): ")
        
        if choice == "1":
            user_id = input("Yangi foydalanuvchi ID sini kiriting: ")
            
            # Ma'lumotlarni yig'ish
            collector = DataCollector()
            success = collector.collect_user_data(user_id)
            
            if success:
                # Encodinglarni yaratish
                encoder = FaceEncoder()
                encoding = encoder.encode_user(user_id)
                
                if encoding is not None:
                    print(f"\nFoydalanuvchi {user_id} muvaffaqiyatli qo'shildi!")
                else:
                    print("\nXatolik: Encodinglarni yaratishda xatolik")
            
        elif choice == "2":
            # Yuz tanib olish tizimini ishga tushirish
            recognition_system = FaceRecognitionSystem()
            recognition_system.run_recognition()
            
        elif choice == "3":
            print("\nDastur tugatildi")
            break
        
        elif choice == "4":
            user_id = input("Foydalanuvchi ID sini kiriting: ")
            encoder = FaceEncoder()  # Encoder obyektini yaratish
            encoder._dalete_encoding(user_id)  # Encodingni o'chirish
            print(f"\nFoydalanuvchi {user_id} muvaffaqiyatli o'chirildi!")

        elif choice == "5":
            encoder = FaceEncoder()  # Encoder obyektini yaratish
            user_ids = encoder.get_all_user_ids()  # Barcha foydalanuvchilarni olish
            print("\nBarcha foydalanuvchilar:")
            for user_id in user_ids:
                print(f"- {user_id}")
        
        else:
            print("\nNoto'g'ri tanlov")

if __name__ == "__main__":
    main()