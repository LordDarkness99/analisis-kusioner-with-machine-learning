# 📊 Analisis Kuesioner dengan Machine Learning

Aplikasi web interaktif berbasis **Streamlit** untuk menganalisis data kuesioner berskala Likert (1–5) secara otomatis.
Aplikasi ini mampu melakukan **preprocessing data**, **clustering responden**, serta **pelatihan model klasifikasi sederhana**.

---

## ✨ Fitur Utama

* **Unggah Data Fleksibel**
  Mendukung format file `.csv`, `.xlsx`, dan `.xls`.

* **Preprocessing Otomatis**
  Menghitung total skor, rata-rata, dan menangani *reversed items* (pertanyaan negatif).

* **Unsupervised Learning (K-Means Clustering)**
  Mengelompokkan responden ke dalam beberapa cluster menggunakan algoritma **K-Means**.

* **Visualisasi Cluster**
  Menampilkan hasil clustering dalam grafik 2D menggunakan **PCA (Principal Component Analysis)**.

* **Supervised Learning (Opsional)**
  Melatih model **Random Forest Classifier** jika data memiliki kolom label/kategori.

* **Unduh Hasil Analisis**
  Mengunduh hasil data yang telah diproses dalam format `.csv`.

---

## ⚙️ Cara Menjalankan Proyek

### 1. Prasyarat

Pastikan Anda telah menginstal:

* **Python 3.8+**
* **pip** (package manager Python)

---

### 2. Instalasi dan Menjalankan Aplikasi

#### a. Clone Repositori

Buka terminal atau Git Bash, lalu jalankan:

```bash
git clone https://github.com/LordDarkness99/analisis-kusioner-with-machine-learning.git
```

#### b. Masuk ke Direktori Proyek, Buat Virtual Environment, Instal Dependensi, dan Jalankan Aplikasi

```bash
# Masuk ke direktori proyek
cd analisis-kusioner-with-machine-learning

# Buat virtual environment (Windows)
python -m venv env
.\env\Scripts\activate

# Buat virtual environment (macOS/Linux)
python3 -m venv env
source env/bin/activate

# Instal dependensi dari requirements.txt
pip install -r requirements.txt

# Jalankan aplikasi Streamlit
streamlit run analisis_kuesioner.py
```

Aplikasi akan otomatis terbuka di browser Anda.

---

## 📄 Format Data Input

File input harus berupa `.csv` atau `.xlsx`.
Setiap baris merepresentasikan satu responden, dan setiap kolom merepresentasikan satu pertanyaan (misalnya `Q1`, `Q2`, `Q3`, dst).
Nilai di setiap kolom harus berupa angka dari **1 hingga 5** (skala Likert).

**Contoh:**

| ID | Q1 | Q2 | Q3 | Q4 |
| -- | -- | -- | -- | -- |
| 1  | 4  | 3  | 5  | 2  |
| 2  | 5  | 4  | 4  | 3  |

---

## 💡 Kontribusi

Silakan buat **pull request** atau ajukan **issue** jika Anda menemukan bug atau ingin menambahkan fitur baru.
