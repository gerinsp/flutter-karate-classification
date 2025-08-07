# 📱 Deteksi gerakan karate 

Deskripsi singkat:

> Aplikasi Flutter untuk klasifikasi gerakan karate, dibuat menggunakan Flutter versi terbaru, mendukung Android dan iOS.

---

## 🛠️ Tools & Teknologi
- Flutter `3.x.x`
- Dart `3.x.x`

---

## 🔧 Cara Clone & Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/gerinsp/flutter-karate-classification.git
   cd flutter-karate-classification
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Jalankan aplikasi**
    - Android / Emulator:
      ```bash
      flutter run
      ```
    - Web (jika diaktifkan):
      ```bash
      flutter run -d chrome
      ```

---

## 📦 Build APK (Android Release)
```bash
flutter build apk --release
```
Hasil build bisa ditemukan di:
```
build/app/outputs/flutter-apk/app-release.apk
```

---

## 📱 Build untuk iOS (Opsional)
> Hanya bisa dijalankan di MacOS dengan Xcode

```bash
flutter build ios
```

---

## 🗂️ Struktur Folder (Opsional)
```
lib/
├── main.dart
├── screens/
├── widgets/
├── models/
├── services/
```

---

## ✍️ Author
- [Gerin_Sena_Pratama](https://github.com/username)

---

## 📃 License
[MIT License](LICENSE)
