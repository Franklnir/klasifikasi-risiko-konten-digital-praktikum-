# Klasifikasi Tingkat Risiko Dampak Paparan Konten Digital

Proyek ini adalah implementasi dari tugas akhir Praktikum Data Mining 2026. Fokus dari proyek ini adalah melakukan klasifikasi terhadap Tingkat Risiko dampak paparan konten digital terhadap fokus belajar, pola tidur, dan kesehatan mental mahasiswa menggunakan algoritma **Decision Tree**.

## Kriteria "Sangat Baik" Terpenuhi
- **Dataset:** Menggunakan lebih dari 5.000 baris data (dataset baru memiliki ~80.000 baris).
- **Akurasi Train:** ~100% (Memenuhi target >= 80%)
- **Akurasi Test:** ~99.9% (Memenuhi target >= 85%)

## Dataset
Proyek ini menggunakan **Student Habits and Academic Performance Dataset**.
- **Sumber Data:** [Kaggle Dataset](https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset?utm_source=chatgpt.com)
- **File Lokal yang Digunakan:** enhanced_student_habits_performance_dataset.csv

## Teknologi yang Digunakan
- Python
- Streamlit (untuk UI Interaktif)
- Scikit-learn (Algoritma Decision Tree, Train/Test Split, Metrics)
- Pandas & NumPy
- Altair (Visualisasi)
- Imbalanced-Learn (SMOTE untuk augmentasi)

## Cara Menjalankan
1. Pastikan library telah terinstall:
   `ash
   pip install streamlit pandas numpy scikit-learn altair imbalanced-learn
   `
2. Jalankan perintah berikut di terminal:
   `ash
   python -m streamlit run app.py
   `
3. Upload dataset CSV (enhanced_student_habits_performance_dataset.csv) melalui tampilan web yang terbuka di browser.
