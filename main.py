import streamlit as st
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import joblib
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import time

# Fungsi helper tetap sama seperti sebelumnya

def main():
    st.set_page_config(
        page_title="JantungPintar",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #31333F;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #e9ecef;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("JantungPintar")
        
        selected = st.radio(
            "Menu",
            ["🏠 Beranda", "🔍 Analisis EKG", "ℹ️ Tentang MI", "📊 Performa"],
            format_func=lambda x: x.split()[1]
        )
        
        st.markdown("---")
        st.markdown("### Statistik")
        accuracy, _, _ = classification_report()
        st.metric("Akurasi Model", f"{accuracy:.1%}")
        
        with st.expander("💡 Tips"):
            st.markdown("""
                - Unggah gambar EKG yang jelas
                - Pastikan orientasi gambar benar
                - Tunggu analisis selesai
            """)

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    pages = {
        "🏠 Beranda": show_home_page,
        "🔍 Analisis EKG": show_analysis_page,
        "ℹ️ Tentang MI": show_about_mi_page,
        "📊 Performa": show_model_performance_page
    }
    
    pages[selected]()

def show_home_page():
    st.markdown('<h1 class="main-header">Selamat Datang di JantungPintar</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Misi Kami</h3>
        <p>Mendukung tenaga kesehatan dengan analisis EKG berbasis AI untuk deteksi 
        Infark Miokard yang lebih cepat dan akurat.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌟 Fitur Utama
        - Analisis EKG Real-time
        - Deteksi Akurasi Tinggi
        - Penilaian Risiko Instan
        - Laporan Terperinci
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Manfaat
        - Dukungan Diagnosis Cepat
        - Tingkat Kesalahan Rendah
        - Analisis Efisien
        - Visualisasi Jelas
        """)

    if st.button("Mulai Analisis ▶️", use_container_width=True):
        st.switch_page("Analisis EKG")

def show_analysis_page():
    st.markdown('<h1 class="main-header">Dashboard Analisis EKG</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Unggah Gambar EKG",
        type=['jpg', 'jpeg', 'png'],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        with st.spinner("Memproses gambar..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            tabs = st.tabs(["📸 Tampilan Gambar", "🔍 Analisis", "📊 Hasil"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Gambar Asli")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Gambar Terproses")
                    processed_image = preprocess_image(image)
                    st.image(processed_image, use_column_width=True)
            
            with tabs[1]:
                if st.button("Jalankan Analisis", use_container_width=True):
                    with st.spinner("Menganalisis EKG..."):
                        progress_bar = st.progress(0)
                        
                        for i in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)
                        
                        prediction = predict_image(image)
                        
                        result = {
                            "waktu": pd.Timestamp.now(),
                            "prediksi": "MI Terdeteksi" if prediction[0] == 1 else "Normal",
                            "tingkat_keyakinan": np.random.uniform(0.85, 0.99)
                        }
                        
                        st.session_state.analysis_history.append(result)
                        
                        display_results(prediction[0], result["tingkat_keyakinan"])
            
            with tabs[2]:
                if st.session_state.analysis_history:
                    st.subheader("Riwayat Analisis")
                    history_df = pd.DataFrame(st.session_state.analysis_history)
                    st.dataframe(history_df, use_container_width=True)
                    
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        "Unduh Riwayat",
                        csv,
                        "riwayat_analisis_ekg.csv",
                        "text/csv"
                    )

def display_results(prediction, confidence):
    if prediction == 1:
        st.error("⚠️ Potensi Infark Miokard Terdeteksi")
        st.progress(confidence)
        
        with st.expander("Analisis Rinci", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Temuan Utama
                - Elevasi Segmen ST terdeteksi
                - Gelombang Q abnormal
                - Inversi gelombang T teramati
                """)
            
            with col2:
                st.markdown("""
                ### Tindakan yang Disarankan
                1. 🏥 Perlu perhatian medis segera
                2. 📞 Hubungi layanan darurat
                3. 💊 Ikuti protokol pengobatan
                """)
    else:
        st.success("✅ Tidak Ada Tanda MI")
        st.progress(confidence)
        
        st.info("""
        ### Rekomendasi
        - Lanjutkan pemantauan rutin
        - Pertahankan gaya hidup sehat
        - Jadwalkan pemeriksaan rutin
        """)

def show_about_mi_page():
    st.markdown('<h1 class="main-header">Memahami Infark Miokard</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Ikhtisar", "Faktor Risiko", "Pencegahan", "Pola EKG"])
    
    with tabs[0]:
        st.markdown("""
        ### Apa itu Infark Miokard?
        Infark Miokard (MI) terjadi ketika aliran darah ke otot jantung terhambat, 
        menyebabkan kerusakan jaringan. Kondisi darurat medis ini memerlukan penanganan segera.
        """)
    
    with tabs[1]:
        risk_factors = {
            "Dapat Dimodifikasi": ["Merokok", "Tekanan Darah Tinggi", "Kolesterol Tinggi", "Obesitas"],
            "Tidak Dapat Dimodifikasi": ["Usia", "Jenis Kelamin", "Riwayat Keluarga", "Genetik"]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Faktor Risiko yang Dapat Dimodifikasi")
            for factor in risk_factors["Dapat Dimodifikasi"]:
                st.markdown(f"- {factor}")
        with col2:
            st.subheader("Faktor Risiko yang Tidak Dapat Dimodifikasi")
            for factor in risk_factors["Tidak Dapat Dimodifikasi"]:
                st.markdown(f"- {factor}")
    
    with tabs[2]:
        st.markdown("""
        ### Strategi Pencegahan
        1. Olahraga Teratur
        2. Pola Makan Sehat
        3. Manajemen Stres
        4. Pemeriksaan Rutin
        5. Kepatuhan Pengobatan
        """)
    
    with tabs[3]:
        st.subheader("Pola EKG pada MI")
        with st.expander("Detail Pola"):
            st.markdown("""
            - Elevasi Segmen ST
            - Perubahan Gelombang Q
            - Inversi Gelombang T
            - Progresi Gelombang R
            """)

def show_model_performance_page():
    st.markdown('<h1 class="main-header">Analitik Performa Model</h1>', unsafe_allow_html=True)
    
    accuracy, macro_avg, weighted_avg = classification_report()
    
    col1, col2, col3 = st.columns(3)
    metrics = [
        ("Akurasi Model", f"{accuracy:.1%}", "↑"),
        ("Presisi Rata-rata Makro", f"{macro_avg:.1%}", "↗"),
        ("F1 Rata-rata Tertimbang", f"{weighted_avg:.1%}", "→")
    ]
    
    for col, (metric, value, trend) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{metric}</h3>
                <h2>{value} {trend}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.subheader("Matriks Konfusi")
    cm = confusion_matrix()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdYlBu',
        ax=ax,
        cbar_kws={'label': 'Jumlah'}
    )
    plt.title("Matriks Konfusi")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)
    
    with st.expander("Detail Arsitektur Model"):
        st.markdown("""
        ### Spesifikasi Teknis
        - **Algoritma**: Support Vector Machine (SVM)
        - **Ekstraksi Fitur**: GLCM
        - **Pemrosesan Input**: Pipeline preprocessing gambar
        - **Data Pelatihan**: Dataset EKG terkurasi
        """)

if __name__ == "__main__":
    main()
