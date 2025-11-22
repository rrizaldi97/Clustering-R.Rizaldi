# 📊 Streamlit Clustering – UTS Pembelajaran Mesin

Aplikasi ini dibuat untuk memenuhi **Ujian Tengah Semester (UTS)** Mata Kuliah *Pembelajaran Mesin*, Semester Ganjil 2025/2026.  
Aplikasi dibangun menggunakan framework **Streamlit** dan mengimplementasikan metode **Clustering (K-Means)** dengan preprocessing & PCA.

---

## 👤 Identitas
| Keterangan | Data |
|-----------|------|
| Nama      | R. Rizaldi |
| NPM       | 072925016 |
| Kelas     | S2 Ilmu Komputer – Semester 1 |
| Topik UTS | Nomor 1 – Clustering |
| Dosen     | Dr. Tjut Awaliyah Zuraiyah, M.Kom |

---

## 🌐 Link Aplikasi (Hosting Streamlit)
Aplikasi bisa diakses langsung melalui link berikut:

👉 **https://clustering-rrizaldi.streamlit.app/**

---

## 📁 Struktur Project
📂 Clustering-R.Rizaldi
- app.py # File utama aplikasi Streamlit
- requirements.txt # Daftar library
- PMA_Investasi.xlsx # Dataset PMA Surabaya
- README.md # Dokumentasi


---

## 📘 Fitur Aplikasi

### 🔧 Preprocessing
- Winsorizing (quantile bawah & atas)
- Normalisasi Z-Score
- Pembersihan data otomatis

### 📉 PCA (Principal Component Analysis)
- Reduksi ke 2 komponen
- Scatter plot interaktif untuk visualisasi pola

### 📊 K-Means Clustering
- Pemilihan jumlah cluster
- Visualisasi cluster secara interaktif
- Ringkasan statistik tiap cluster

### 🔮 Prediksi Cluster Baru
Input 4 variabel:
- Nilai Investasi  
- Jumlah Proyek  
- TKI  
- TKA  

Output berupa:
- Cluster prediksi  
- Interpretasi cluster  

### 📈 Visualisasi
- Top 10 Negara berdasarkan nilai investasi
- Pie chart pangsa investasi
- PCA scatter plot
- Grafik Elbow & Silhouette

---

## 📎 Dataset
Dataset tugas UTS:  
**PMA_Investasi.xlsx** (dari folder Clustering – Nomor 1 pada Google Drive dosen).

---

## 📝 Catatan
Project ini hanya digunakan untuk keperluan akademik UTS dan tidak untuk distribusi komersial.

---

## 🙏 Terima Kasih
Terima kasih kepada dosen pengampu serta rekan-rekan yang telah memberikan arahan dalam pengerjaan aplikasi ini.





