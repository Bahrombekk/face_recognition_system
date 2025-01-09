# Face Recognition System - O'rnatish va Ishlatish Qo'llanmasi

## Loyiha strukturasi

```
face_recognition_system/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── face_recognition_system.py
│   ├── face_processor.py
│   ├── face_encoder.py
│   ├── data_collector.py
│   ├── database.py
│   ├── clustering.py
│   └── config.py
├── dataset/
├── database/
├── logs/
├── requirements.txt
└── README.md
```

## O'rnatish

1. Python 3.8 yoki undan yuqori versiyasini o'rnating

2. Virtual muhit yarating va aktivlashtiring:
```bash
python -m venv venv
# Windows uchun
venv\Scripts\activate
# Linux/Mac uchun
source venv/bin/activate
```

3. Requirements o'rnating:
```bash
pip install -r requirements.txt
```

4. Kerakli papkalarni yarating:
```bash
mkdir dataset database logs
```

## Ishlatish

1. Dasturni ishga tushirish:
```bash
python src/main.py
```

2. Asosiy menyudan quyidagi imkoniyatlardan foydalanishingiz mumkin:
   - 1: Yangi foydalanuvchi qo'shish
   - 2: Yuzni tanib olishni boshlash
   - 3: Chiqish
   - 4: Foydalanuvchini o'chirish
   - 5: Barcha foydalanuvchilarni ko'rish

### Yangi foydalanuvchi qo'shish:
1. Menyudan "1" ni tanlang
2. Foydalanuvchi ID sini kiriting
3. Kamera ochilganda, quyidagi holatlarni yozib olish kerak:
   - To'g'ridan qarab
   - Chap tomondan
   - O'ng tomondan
   - Tabassum bilan
   - Turli yorug'lik sharoitlarida

### Yuz tanib olishni boshlash:
1. Menyudan "2" ni tanlang
2. Kamera ochiladi
3. Chiqish uchun 'q' tugmasini bosing

## Muhim eslatmalar:
- Yaxshi yoritilgan joyda ishlatish tavsiya etiladi
- Yuz ro'yxatga olishda turli burchaklar va holatlardan foydalaning
- Kamera yuqori sifatli bo'lgani ma'qul# face_recognition_system
