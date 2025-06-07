# Menyelesaikan Permasalahan Institusi Pendidikan - Bintang Cahya Anwar

## Business Understanding

Jaya Jaya Institute merupakan lembaga pendidikan tinggi yang telah berdiri sejak tahun 2000 dan dikenal luas karena berhasil meluluskan banyak alumni berprestasi. Namun, di balik pencapaian tersebut, institusi ini juga menghadapi tantangan serius berupa tingginya angka mahasiswa yang tidak menyelesaikan studinya (dropout).

Fenomena dropout ini menjadi perhatian utama karena dapat memengaruhi reputasi institusi serta efektivitas sistem pendidikan yang dijalankan. Untuk mengantisipasi hal tersebut, Jaya Jaya Institute berinisiatif mengembangkan sistem prediktif yang mampu mengidentifikasi mahasiswa yang berisiko dropout sejak dini, agar dapat diberikan intervensi dan pendampingan yang tepat.

### Permasalahan Bisnis
- Tingginya tingkat putus studi (dropout) di institusi pendidikan berdampak negatif pada reputasi, akreditasi, dan pendanaan lembaga.
- Tidak adanya sistem deteksi dini untuk mengidentifikasi mahasiswa yang berisiko tinggi untuk putus studi, sehingga intervensi pencegahan sulit dilakukan secara tepat waktu.
- Kurangnya pemahaman berbasis data tentang faktor-faktor utama penyebab dropout, seperti prestasi akademik, kondisi sosial ekonomi, kehadiran, atau keterlibatan kampus.
- Kerugian jangka panjang baik bagi institusi maupun mahasiswa, termasuk rendahnya tingkat kelulusan, pemborosan sumber daya, dan meningkatnya beban administratif.
- Kebutuhan akan pendekatan prediktif berbasis machine learning untuk mendukung pengambilan keputusan yang lebih cepat, akurat, dan proaktif dalam manajemen risiko dropout.

### Cakupan Proyek

Proyek ini mencakup beberapa tahapan kunci untuk mencapai tujuan analisis dan prediksi dropout mahasiswa di Jaya Jaya Institute, yaitu:

1. **Eksplorasi Data**  
   Melakukan pengumpulan data mahasiswa serta analisis awal untuk memahami pola dropout dan faktor-faktor yang memengaruhinya.
2. **Pra-pemrosesan Data**  
   Menyiapkan data dengan langkah-langkah seperti pengisian nilai yang hilang, encoding variabel kategorikal, dan normalisasi fitur untuk memastikan kualitas data yang optimal.
3. **Pengembangan Model Machine Learning**  
   Membangun model machine learning untuk memprediksi kemungkinan mahasiswa tidak menyelesaikan studi berdasarkan data historis yang telah diproses.
4. **Evaluasi Model**  
   Mengukur performa model menggunakan berbagai metrik evaluasi, termasuk akurasi, precision, recall, dan F1-score, untuk memastikan hasil yang akurat.  
5. **Pembuatan Dashboard**  
   Mengembangkan dashboard untuk menyajikan insight yang diperoleh dari hasil analisis data dan prediksi model secara lebih interaktif dan mudah dipahami oleh pihak manajemen institusi.
6. **Rekomendasi Action**  
   Menyusun rekomendasi berdasarkan hasil analisis untuk membantu institusi dalam mengelola dan menurunkan tingkat dropout secara lebih proaktif.
7. **Dokumentasi**  
   Menyusun laporan proyek yang komprehensif untuk memastikan setiap langkah dalam proses analisis terdokumentasi dengan baik.

### Persiapan

#### Sumber Data
Dataset yang digunakan dalam proyek ini berasal dari Github Dicoding Academy, yang berisi informasi terkait data mahasiswa dari institusi fiktif "Jaya Jaya Institute". Dataset ini bertujuan untuk mendukung pembelajaran analisis data di bidang pendidikan tinggi dan digunakan untuk memprediksi dropout atau ketidaksanggupan mahasiswa dalam menyelesaikan studinya. Dataset dapat diakses melalui tautan berikut: [Jaya Jaya Institute](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv). 

Dataset ini memuat berbagai informasi penting mengenai mahasiswa, seperti:
- `Marital_status` (int): Status pernikahan mahasiswa (1 = Single, 2 = Married, 3 = Widower, dst).
- `Application_mode` (int): Mode pendaftaran yang dipilih mahasiswa (1 = 1st phase - general contingent, 2 = Ordinance No. 612/93, 5 = 1st phase - special contingent (Azores Island), dst).
- `Application_order` (int): Urutan pesanan aplikasi (0 - 9).
- `Course` (int): Program studi yang diambil mahasiswa (33 = Biofuel Production Technologies, 171 = Animation and Multimedia Design, dst).
- `Daytime_evening_attendance` (int): Jenis kehadiran (1 = daytime, 0 = evening).
- `Previous_qualification` (int): Jenis kualifikasi pendidikan sebelumnya (1 = Secondary education, 2 = Higher education - bachelor's degree, 3 = Higher education - degree, dst).
- `Previous_qualification_grade` (float): Nilai rata-rata dari pendidikan sebelumnya (0.0 - 200.00).
- `Nacionality` (int): Kewarganegaraan mahasiswa (1 = Portuguese, 2 = German, 6 = Spanish, dst).
- `Mothers_qualification` (int): Tingkat pendidikan ibu mahasiswa (1 = Secondary Education - 12th Year of Schooling or Eq, 2 = Higher Education - Bachelor's Degree, dst).
- `Fathers_qualification` (int): Tingkat pendidikan ayah mahasiswa (1 = Secondary Education - 12th Year of Schooling or Eq, 2 = Higher Education - Bachelor's Degree, dst).
- `Mothers_occupation` (int): Jenis pekerjaan ibu (0 = Student, 1 = Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers, dst).
- `Fathers_occupation` (int): Jenis pekerjaan ayah (0 = Student, 1 = Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers, dst).
- `Admission_grade` (float): Nilai ujian masuk mahasiswa (0 - 200).
- `Displaced` (int): Menunjukkan apakah mahasiswa berasal dari daerah yang terkena dampak sosial/geografis tertentu (0 = No, 1 = Yes).
- `Educational_special_needs` (int): Menunjukkan apakah mahasiswa memiliki kebutuhan pendidikan khusus ((0 = No, 1 = Yes)).
- `Debtor` (int): Status utang pendidikan (0 = No, 1 = Yes).
- `Tuition_fees_up_to_date` (int): Status pembayaran biaya kuliah (0 = No, 1 = Yes).
- `Gender` (int): Jenis kelamin mahasiswa (1 = Male, 0 = Female).
- `Scholarship_holder` (int): Menunjukkan apakah mahasiswa penerima beasiswa (0 = No, 1 = Yes).
- `Age_at_enrollment` (int): Usia saat mahasiswa mulai kuliah.
- `International` (int): Menunjukkan apakah mahasiswa berasal dari luar negeri (0 = No, 1 = Yes).
- `Curricular_units_1st_sem_credited` (int): Jumlah SKS yang dikonversi/diakui di semester 1.
- `Curricular_units_1st_sem_enrolled` (int): Jumlah mata kuliah yang diambil di semester 1.
- `Curricular_units_1st_sem_evaluations` (int): Jumlah evaluasi/ujian yang diikuti di semester 1.
- `Curricular_units_1st_sem_approved` (int): Jumlah mata kuliah yang lulus di semester 1.
- `Curricular_units_1st_sem_grade` (float): Rata-rata nilai di semester 1 (0 - 20).
- `Curricular_units_1st_sem_without_evaluations` (int): Jumlah mata kuliah tanpa evaluasi di semester 1.
- `Curricular_units_2nd_sem_credited` (int): Jumlah SKS yang dikonversi/diakui di semester 2.
- `Curricular_units_2nd_sem_enrolled` (int): Jumlah mata kuliah yang diambil di semester 2.
- `Curricular_units_2nd_sem_evaluations` (int): Jumlah evaluasi/ujian yang diikuti di semester 2.
- `Curricular_units_2nd_sem_approved` (int): Jumlah mata kuliah yang lulus di semester 2.
- `Curricular_units_2nd_sem_grade` (float): Rata-rata nilai di semester 2 (0 - 20).
- `Curricular_units_2nd_sem_without_evaluations` (int): Jumlah mata kuliah tanpa evaluasi di semester 2.
- `Unemployment_rate` (float): Tingkat pengangguran nasional pada saat pendaftaran.
- `Inflation_rate` (float): Tingkat inflasi nasional pada saat pendaftaran.
- `GDP` (float): Produk Domestik Bruto (GDP) negara pada saat pendaftaran.
- `Status` (object): Status akhir mahasiswa (target prediksi), seperti Graduate, Dropout, atau Enrolled.

Dengan informasi ini, kita bisa memahami tipe data untuk setiap fitur dan memutuskan langkah-langkah selanjutnya dalam pemrosesan dan pembersihan data.

#### Setup Environment
Sebelum menjalankan proyek ini, penting untuk menyiapkan lingkungan kerja yang terisolasi (virtual environment). Ini berguna agar dependensi dan versi library yang digunakan proyek tidak bertabrakan dengan aplikasi lain di komputer Anda. Pada proyek ini, Anda bisa memilih menggunakan Anaconda atau pipenv sesuai preferensi.
##### 1. Menggunakan Anaconda
Anaconda adalah distribusi Python yang populer dan memudahkan manajemen environment serta paket.
- Buat environment baru dengan versi Python 3.9 (disarankan agar sesuai dengan pengembangan proyek).
- Aktifkan environment untuk mulai menggunakan lingkungan tersebut.
- Instal dependensi dari file requirements.txt agar semua library yang dibutuhkan tersedia.
```bash
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

##### 2. Menggunakan Shell/Terminal dengan pipenv
pipenv adalah alat untuk membuat virtual environment dan mengelola dependensi Python secara otomatis.
- Pasang pipenv jika belum tersedia.
- Buat environment dan instal semua library yang tercantum di Pipfile (jika ada) atau langsung instal dari requirements.txt.
- Aktifkan shell virtual environment agar perintah Python yang dijalankan berada di dalam lingkungan tersebut.
```bash
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

#### Menjalankan Proyek
Setelah environment siap dan semua dependensi terinstal, Anda bisa mulai menjalankan proyek dengan langkah-langkah berikut:
##### Menjalankan Notebook
- Buka folder proyek di code editor (misal VSCode atau Jupyter Notebook).
- Jalankan notebook (.ipynb) untuk melakukan eksplorasi data, preprocessing, dan pengembangan model machine learning.
- Pastikan environment yang sudah dibuat tadi sudah aktif saat menjalankan kode agar library yang diperlukan tersedia.

#### Menjalankan Aplikasi Streamlit (Student Dropout Prediction)
Aplikasi Student Dropout Prediction dapat dijalankan baik secara **lokal** maupun melalui versi **online**. Aplikasi ini dirancang untuk memprediksi kemungkinan mahasiswa mengalami dropout, berdasarkan data yang diinput secara interaktif melalui antarmuka web yang sederhana dan intuitif.

##### Menjalankan Secara Lokal
- Pastikan environment aktif dan seluruh dependensi telah diinstal (melalui requirements.txt).
- Jalankan aplikasi dengan perintah berikut di terminal:
```bash
streamlit run app.py
```
- Setelah dijalankan, aplikasi akan dapat diakses melalui browser di alamat lokal berikut:
```bash
http://localhost:8501
```

##### Mengakses Versi Online
Selain dijalankan secara lokal, aplikasi ini juga telah dideploy secara online dan dapat diakses langsung melalui tautan berikut:
[Student Dropout Risk Analysis](https://education-issues.streamlit.app/)

## Business Dashboard

### Penjelasan Dashboard
Dashboard ini memberikan visualisasi terkait performa akademik mahasiswa di Jaya Jaya Institute, termasuk faktor-faktor yang memengaruhi keberhasilan studi, statistik peserta didik, serta prediksi risiko dropout mahasiswa. Dashboard dikembangkan menggunakan Metabase agar mudah diakses dan digunakan oleh tim akademik maupun manajemen institusi.

### Cara Mengakses Dashboard
Dashboard dapat diakses melalui platform Metabase dengan cara berikut:
1. Buka browser dan kunjungi alamat URL berikut: [Metabase Dashboard](http://localhost:3000/).
2. Gunakan akun berikut untuk masuk:
   - Email: root123@mail.com
   - Password: root123
3. Setelah berhasil login, Anda akan langsung menuju halaman utama dashboard. Pilih dashboard yang diinginkan untuk melihat berbagai visualisasi data yang tersedia.

### Contoh Tampilan Dashboard
Berikut adalah contoh tampilan dari dashboard yang telah dikembangkan:

![Metabase Dashboard](https://github.com/bintang58/education-issues/blob/main/bintangcahya58-dashboard.png?raw=true)

Dashboard ini dirancang untuk memudahkan eksplorasi data performa akademik mahasiswa serta memberikan insight penting yang membantu pihak institusi dalam mengambil keputusan strategis guna meningkatkan kualitas pendidikan dan menurunkan angka dropout.

Visualisasi yang tersedia mencakup berbagai aspek penting seperti distribusi nilai akademik berdasarkan jurusan, tingkat kehadiran, keterlibatan dalam kegiatan belajar, serta prediksi risiko dropout berdasarkan model machine learning.

Dengan dashboard ini, tim akademik dan manajemen dapat memantau tren performa mahasiswa secara real-time sehingga langkah-langkah intervensi dapat diambil secara tepat waktu untuk mendukung kesuksesan studi mahasiswa.

## Evaluasi Model
Setelah tahap pemrosesan data dan pengembangan model selesai, langkah berikutnya adalah mengevaluasi performa model dalam memprediksi status mahasiswa, apakah berisiko dropout atau tidak.

Model yang digunakan adalah **XGBoost**, yang telah dioptimasi dengan **hyperparameter tuning** menggunakan **Grid Search** untuk menemukan kombinasi parameter terbaik agar performa model maksimal dan mampu melakukan generalisasi dengan baik pada data baru.

Evaluasi model dilakukan dengan beberapa metrik kinerja berikut:
- Accuracy: Persentase prediksi yang benar secara keseluruhan dalam mengklasifikasikan mahasiswa yang dropout dan tidak dropout.
- Precision: Proporsi prediksi mahasiswa yang diprediksi dropout yang benar-benar mengalami dropout, mengukur tingkat false positive.
- Recall: Kemampuan model untuk mendeteksi seluruh mahasiswa yang benar-benar mengalami dropout (sensitivitas).
- F1-Score: Kombinasi harmonik dari precision dan recall, memberikan gambaran keseimbangan antara keduanya, sangat penting terutama jika kelas dropout dan tidak dropout tidak seimbang.

Hasil evaluasi model adalah sebagai berikut:
- Accuracy: 0.86
- F1-Score: 0.86
- Precision: 0.86
- Recall: 0.86

**Classification Report**:
```plaintext
                   precision    recall  f1-score   support

Enrolled/Graduate       0.87      0.93      0.90       569
          Dropout       0.85      0.74      0.79       316

         accuracy                           0.86       885
        macro avg       0.86      0.83      0.84       885
     weighted avg       0.86      0.86      0.86       885
```

Model mencapai akurasi 86% dengan F1-score yang seimbang, dimana precision dan recall untuk kelas dropout masing-masing sebesar 85% dan 74%. Hal ini menunjukkan model sudah cukup akurat secara keseluruhan, namun kemampuan mendeteksi mahasiswa yang berisiko dropout masih perlu ditingkatkan. Recall yang lebih tinggi penting agar risiko dropout dapat teridentifikasi lebih baik. Oleh karena itu, disarankan melakukan penanganan ketidakseimbangan kelas dan tuning model lebih lanjut untuk meningkatkan sensitivitas dan performa prediksi model.

## Conclusion
Berdasarkan analisis data dan evaluasi model prediksi putus studi, berikut beberapa temuan utama:
- Tingginya tingkat dropout ditemukan terutama pada kelompok mahasiswa dengan IPK rendah, tingkat kehadiran yang buruk, dan kurangnya keterlibatan dalam kegiatan akademik.
- Faktor-faktor utama yang berkontribusi terhadap risiko dropout meliputi prestasi akademik, dukungan sosial, kondisi ekonomi, serta aktivitas non-akademik.
- Model machine learning yang dikembangkan menunjukkan akurasi prediksi yang baik, namun nilai recall masih perlu ditingkatkan agar lebih efektif dalam mengidentifikasi mahasiswa yang benar-benar berisiko tinggi untuk putus studi.
- Dashboard interaktif yang dibangun berhasil memberikan insight visual yang berguna bagi pihak manajemen akademik untuk memahami pola dropout secara lebih mendalam dan mengambil tindakan yang berbasis data.

## Rekomendasi Action Items
Untuk menurunkan tingkat dropout secara efektif, institusi pendidikan dapat mempertimbangkan langkah-langkah berikut:
- Perkuat sistem pemantauan akademik dan kehadiran untuk mendeteksi mahasiswa yang menunjukkan tanda-tanda awal risiko dropout.
- Tingkatkan dukungan akademik dan emosional melalui program mentoring, konseling, dan bantuan belajar bagi mahasiswa yang berjuang secara akademik atau sosial.
- Sediakan bantuan finansial dan beasiswa bagi mahasiswa dari latar belakang ekonomi lemah, agar mereka tidak terpaksa berhenti kuliah karena kendala biaya.
- Optimalkan pemanfaatan model prediksi dalam sistem akademik untuk memberikan peringatan dini kepada dosen wali atau staf konselor.
- Fasilitasi keterlibatan mahasiswa dalam kegiatan ekstrakurikuler dan komunitas kampus untuk meningkatkan rasa memiliki dan keterikatan terhadap institusi.
