# kuesioner_analysis_streamlit.py
# Streamlit app for automated analysis of Likert-scale questionnaire + simple ML dashboard
# Features:
# - Upload CSV/Excel of responses (expected: first column optional ID, next columns Q1..Qn with values 1-5)
# - Specify reversed items (e.g. Q2,Q4,Q6) -- app will flip those automatically
# - Compute per-respondent total & average scores, assign categorical labels
# - KMeans clustering (interactive n_clusters)
# - PCA 2D visualization of clusters
# - If user provides an existing label column, can train a RandomForest classifier and show metrics
# - Download processed dataset (with flipped items, scores, clusters, categories)

import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Kuesioner Analyzer', layout='wide')

# -------------------------- Helper functions --------------------------

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None


def detect_question_columns(df):
    # Heuristic: columns that start with Q or have numeric-like values 1-5
    qcols = [c for c in df.columns if str(c).strip().upper().startswith('Q')]
    if len(qcols) >= 1:
        return qcols
    # fallback: choose columns with mostly integers 1-5
    candidate = []
    for c in df.columns:
        ser = df[c].dropna()
        if pd.api.types.is_numeric_dtype(ser):
            unique = ser.unique()
            if all((u in [1,2,3,4,5] for u in unique)):
                candidate.append(c)
    return candidate


def flip_reversed(df, reversed_items):
    # Flip scale 1-5 by using 6 - x
    for col in reversed_items:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 6 - x if pd.notnull(x) else x)
    return df


def compute_scores(df, qcols):
    df['total_score'] = df[qcols].sum(axis=1)
    df['average_score'] = df[qcols].mean(axis=1)
    return df


def categorize(avg):
    # You can tweak these boundaries
    if avg >= 4.1:
        return 'Sangat Baik'
    elif avg >= 3.1:
        return 'Baik'
    elif avg >= 2.1:
        return 'Cukup'
    elif avg >= 1.1:
        return 'Kurang'
    else:
        return 'Sangat Kurang'

# -------------------------- App UI --------------------------

st.title('Kuesioner Analyzer — Likert (1–5)')
st.markdown(
    """
    Aplikasi ini otomatis memproses data kuesioner skala Likert (1-5).
    - Unggah file CSV atau Excel (kolom: ID (opsional), Q1..Qn)
    - Masukkan daftar item *reversed* (misal: Q2,Q4,Q6)
    - Lihat hasil preprocessing, clustering, dan (opsional) klasifikasi
    """
)

with st.sidebar:
    st.header('Pengaturan')
    uploaded_file = st.file_uploader('Unggah CSV atau Excel (Q1..Qn)', type=['csv','xlsx','xls'])
    reversed_input = st.text_input('Daftar item reversed (pisahkan dengan koma)', value='Q2,Q4,Q6')
    run_clustering = st.checkbox('Jalankan clustering (KMeans)', value=True)
    n_clusters = st.number_input('Jumlah cluster (KMeans)', min_value=2, max_value=10, value=3)
    run_classifier = st.checkbox('Latih classifier jika ada kolom label', value=False)
    label_column = st.text_input('Nama kolom label (jika ada, misal: kategori)', value='')
    random_state = st.number_input('Random state (untuk reproduksibilitas)', value=42)
    st.markdown('---')
    st.markdown('Template CSV: kolom Q1..Qn harus berisi nilai 1-5 (SS..STS).')
    if st.button('Download template contoh'):
        sample = pd.DataFrame({f'Q{i}': [5,4,3] for i in range(1,11)})
        buf = io.BytesIO()
        sample.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button('Klik untuk unduh template CSV', data=buf, file_name='template_kuesioner.csv', mime='text/csv')

if uploaded_file is None:
    st.info('Silakan unggah file kuesioner (.csv / .xlsx) di sidebar untuk memulai.')
    st.stop()

# Load data
df = load_data(uploaded_file)
if df is None:
    st.stop()

st.subheader('Preview data (5 baris pertama)')
st.dataframe(df.head())

# Detect question columns
qcols = detect_question_columns(df)
if len(qcols) == 0:
    st.error('Tidak menemukan kolom pertanyaan otomatis. Pastikan ada kolom bernama Q1..Qn atau kolom berisi angka 1-5.')
    st.stop()

st.write('Kolom pertanyaan terdeteksi:', qcols)

# Parse reversed items from input
reversed_items = [s.strip() for s in reversed_input.split(',') if s.strip()]
st.write('Reversed items:', reversed_items)

# Ensure questions are numeric and in 1-5 range
for c in qcols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Flip reversed
df_proc = df.copy()
df_proc = flip_reversed(df_proc, reversed_items)

# Compute scores
df_proc = compute_scores(df_proc, qcols)

# Categorize
df_proc['kategori'] = df_proc['average_score'].apply(categorize)

st.subheader('Hasil preprocessing')
st.write('Beberapa statistik:')
st.write(df_proc[['total_score','average_score','kategori']].describe())

st.dataframe(df_proc.head())

# Download processed data
csv_buf = df_proc.to_csv(index=False).encode('utf-8')
st.download_button('Unduh dataset yang sudah diproses (CSV)', data=csv_buf, file_name='kuesioner_processed.csv', mime='text/csv')

# -------------------------- Clustering --------------------------
if run_clustering:
    st.subheader('Clustering: KMeans')
    X = df_proc[qcols].fillna(df_proc[qcols].mean())
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state))
    clusters = kmeans.fit_predict(X)
    df_proc['cluster'] = clusters

    st.write('Jumlah tiap cluster:')
    st.write(df_proc['cluster'].value_counts().sort_index())

    # PCA 2D visualization
    pca = PCA(n_components=2, random_state=int(random_state))
    Xr = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=Xr[:,0], y=Xr[:,1], hue=clusters, palette='deep', ax=ax)
    ax.set_title('PCA 2D visualization of clusters')
    st.pyplot(fig)

    st.dataframe(df_proc.groupby('cluster')[qcols].mean().round(2))

# -------------------------- Classifier (optional) --------------------------
if run_classifier and label_column.strip() != '' and label_column in df_proc.columns:
    st.subheader('Training classifier (RandomForest)')
    # Prepare data
    X = df_proc[qcols].fillna(df_proc[qcols].mean())
    y = df_proc[label_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(random_state), stratify=y if len(y.unique())>1 else None)
    clf = RandomForestClassifier(n_estimators=100, random_state=int(random_state))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write('Classification report:')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred, labels=list(np.unique(y)))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    st.pyplot(fig2)

elif run_classifier and (label_column.strip() == '' or label_column not in df_proc.columns):
    st.info('Tidak ada kolom label ditemukan atau nama kolom kosong. Jika ingin pakai classifier, masukkan nama kolom label yang ada di dataset.')

# -------------------------- Aggregated Visuals --------------------------
st.subheader('Visualisasi ringkas')
col1, col2 = st.columns(2)
with col1:
    st.write('Distribusi kategori')
    fig3, ax3 = plt.subplots()
    sns.countplot(x='kategori', data=df_proc, order=['Sangat Baik','Baik','Cukup','Kurang','Sangat Kurang'], ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with col2:
    st.write('Boxplot skor per cluster (jika ada)')
    if 'cluster' in df_proc.columns:
        fig4, ax4 = plt.subplots()
        sns.boxplot(x='cluster', y='average_score', data=df_proc, ax=ax4)
        st.pyplot(fig4)
    else:
        st.write('Cluster belum dijalankan.')

st.info('Selesai — kamu dapat menyesuaikan pengaturan di sidebar (reversed items, jumlah cluster, label column).')

# Footer
st.markdown('---')
st.markdown('**Catatan:** aplikasi ini memakai asumsi skala 1–5. Pastikan data input sesuai. Kamu bisa menghubungi pembuat untuk penyesuaian lebih lanjut.')
