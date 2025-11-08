import streamlit as st
import pandas as pd
import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# IMPORTS YANG BENAR UNTUK LANGCHAIN==0.1.20
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import List, Dict, Any
import json
import re # For parsing tool queries


# --- Konfigurasi Awal Streamlit ---
st.set_page_config(page_title="👨‍🔬 Asisten Data Scientist Pro", layout="wide")
st.title("👨‍🔬 Asisten Data Scientist Profesional")
st.markdown("""
Selamat datang! Saya adalah asisten cerdas Anda untuk analisis data dan penulisan ilmiah.
Unggah dataset Anda dan ajukan pertanyaan untuk mendapatkan wawasan statistik, interpretasi, dan saran penulisan akademik.
""")

# Inisialisasi state sesi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = ""
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "_last_api_key_used" not in st.session_state:
    st.session_state._last_api_key_used = None
if "google_api_key_input_value" not in st.session_state:
    st.session_state.google_api_key_input_value = ""


# --- Definisi Tools untuk Agen ---
@tool
def context_retriever_tool(query: str) -> str:
    """
    Mengambil konteks dan informasi relevan dari analisis atau dokumen sebelumnya yang telah disimpan.
    Gunakan ini untuk mendapatkan data historis atau interpretasi yang relevan dari vector store.
    Input adalah query string yang menjelaskan informasi yang dicari.
    """
    if st.session_state.vector_store:
        try:
            docs = st.session_state.vector_store.similarity_search(query, k=5)
            return "Retrieved Context:\n" + "\n---\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Error retrieving context: {e}"
    return "Tidak ada konteks yang tersimpan atau vector store belum diinisialisasi."

@tool
def dataframe_inspector_tool(command: str = "general_info") -> str:
    """
    Menginspeksi dataframe yang sedang diunggah untuk mendapatkan informasi struktur atau statistik dasar.
    Parameter `command` dapat berupa:
    - 'general_info': Untuk ringkasan umum (shape, columns, dtypes, head).
    - 'describe [column_name]': Untuk mendapatkan statistik deskriptif kolom spesifik (misal: 'describe usia').
    - 'value_counts [column_name]': Untuk mendapatkan frekuensi nilai unik kolom kategorikal (misal: 'value_counts pendidikan').
    - 'correlation [col1] [col2]': Untuk menghitung korelasi Pearson antara dua kolom numerik.
    - 'mean [column_name]': Untuk menghitung rata-rata kolom numerik.
    - 'median [column_name]': Untuk menghitung median kolom numerik.
    - 'std [column_name]': Untuk menghitung standar deviasi kolom numerik.
    """
    df = st.session_state.dataframe
    if df is None:
        return "Dataset belum diunggah."

    if command == "general_info":
        info = f"Dataset Shape: {df.shape}\n"
        info += f"Columns: {df.columns.tolist()}\n"
        info += "Dtypes:\n"
        info += df.dtypes.to_markdown() + "\n"
        info += "First 5 rows:\n"
        info += df.head().to_markdown(index=False)
        return info
    
    match_describe = re.match(r"describe\s+([a-zA-Z0-9_]+)", command)
    match_value_counts = re.match(r"value_counts\s+([a-zA-Z0-9_]+)", command)
    match_correlation = re.match(r"correlation\s+([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)", command)
    match_mean = re.match(r"mean\s+([a-zA-Z0-9_]+)", command)
    match_median = re.match(r"median\s+([a-zA-Z0-9_]+)", command)
    match_std = re.match(r"std\s+([a-zA-Z0-9_]+)", command)

    if match_describe:
        column_name = match_describe.group(1)
        if column_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[column_name]):
                return f"Descriptive stats for '{column_name}':\n" + df[column_name].describe().to_markdown()
            else:
                return f"Kolom '{column_name}' bukan numerik. Unique values: {df[column_name].nunique()}, Top 5 values:\n{df[column_name].value_counts().head(5).to_markdown()}"
        else:
            return f"Kolom '{column_name}' tidak ditemukan di dataset."
    elif match_value_counts:
        column_name = match_value_counts.group(1)
        if column_name in df.columns:
            return f"Value counts for '{column_name}':\n" + df[column_name].value_counts().to_markdown()
        else:
            return f"Kolom '{column_name}' tidak ditemukan di dataset."
    elif match_correlation:
        col1, col2 = match_correlation.groups()
        if col1 in df.columns and col2 in df.columns:
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                correlation = df[[col1, col2]].corr().loc[col1, col2]
                return f"Korelasi Pearson antara '{col1}' dan '{col2}': {correlation:.4f}"
            else:
                return f"Salah satu atau kedua kolom ('{col1}', '{col2}') bukan numerik."
        else:
            return f"Kolom '{col1}' atau '{col2}' tidak ditemukan."
    elif match_mean:
        column_name = match_mean.group(1)
        if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
            return f"Rata-rata kolom '{column_name}': {df[column_name].mean():.4f}"
        else:
            return f"Kolom '{column_name}' tidak ditemukan atau bukan numerik."
    elif match_median:
        column_name = match_median.group(1)
        if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
            return f"Median kolom '{column_name}': {df[column_name].median():.4f}"
        else:
            return f"Kolom '{column_name}' tidak ditemukan atau bukan numerik."
    elif match_std:
        column_name = match_std.group(1)
        if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
            return f"Standar deviasi kolom '{column_name}': {df[column_name].std():.4f}"
        else:
            return f"Kolom '{column_name}' tidak ditemukan atau bukan numerik."
    else:
        return "Perintah inspeksi tidak valid atau format tidak sesuai. Lihat deskripsi tool untuk format yang benar."

# Daftar tools yang akan digunakan oleh agen
TOOLS_FOR_AGENT = [context_retriever_tool, dataframe_inspector_tool]


# --- Sidebar untuk unggah file dan instruksi ---
with st.sidebar:
    st.header("⚙️ Konfigurasi & Data")

    # Input Google API Key
    st.subheader("🔑 Pengaturan Google API Key")
    google_api_key_input = st.text_input(
        "Masukkan Google API Key Anda",
        type="password",
        value=st.session_state.google_api_key_input_value,
        help="Dapatkan API Key dari Google AI Studio: https://makersuite.google.com/app/apikey"
    )
    st.session_state.google_api_key_input_value = google_api_key_input

    # Tombol reset aplikasi
    if st.button("Reset Aplikasi", help="Membersihkan semua data yang diunggah dan riwayat chat."):
        for key in list(st.session_state.keys()):
            if key != "google_api_key_input_value":
                del st.session_state[key]
        st.info("Aplikasi telah direset.")
        st.rerun()

    uploaded_file = st.file_uploader("Unggah Dataset (CSV)", type=["csv"])
    st.markdown("""
    ---
    **Cara Menggunakan:**
    1.  Masukkan Google API Key Anda di atas.
    2.  Unggah file CSV Anda.
    3.  Dataset akan dianalisis deskriptif secara otomatis.
    4.  Ajukan pertanyaan di kolom chat di bawah.
    5.  Contoh pertanyaan:
        -   "Buatkan ringkasan statistik deskriptif dan interpretasinya."
        -   "Bagaimana korelasi antara [kolom A] dan [kolom B]?"
        -   "Tuliskan bagian metodologi untuk analisis regresi dengan variabel dependen [kolom Y] dan independen [kolom X]."
        -   "Saran visualisasi untuk melihat distribusi [kolom A]?"
        -   "Buatkan bagian abstrak untuk penelitian ini."
    """)
    st.info("Pastikan kolom numerik Anda bersih dari nilai non-numerik untuk analisis statistik yang akurat.")

google_api_key = google_api_key_input

# --- Pengecekan awal API Key dan Penghentian jika tidak ada ---
if not google_api_key:
    st.info("Silakan masukkan Google API Key Anda di sidebar untuk melanjutkan.", icon="🗝️")
    st.stop()

# --- Inisialisasi LLM, Embeddings, dan Agen (dilakukan sekali atau saat API Key berubah) ---
if (st.session_state.llm is None or st.session_state.embeddings is None or st.session_state.agent_executor is None) or \
   (st.session_state._last_api_key_used != google_api_key):
    
    if st.session_state._last_api_key_used is not None and st.session_state._last_api_key_used != google_api_key:
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.dataframe = None
        st.session_state.analysis_summary = ""
        st.session_state.uploaded_file_name = None
        st.info("API Key diperbarui. Riwayat chat dan data telah direset.")

    try:
        # --- Solusi untuk "No current event loop" ---
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        # --- Akhir solusi ---

        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=google_api_key)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        
        # --- Agent Initialization ---
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "Anda adalah seorang Data Scientist profesional, ahli di bidang data analytics, machine learning, dan penulisan ilmiah. "
                        "Tugas Anda adalah menganalisis data, menjelaskan insight statistik, serta menulis bagian ilmiah seperti abstrak, metodologi, hasil, pembahasan, dan kesimpulan. "
                        "Gunakan gaya bahasa ilmiah yang formal, logis, dan argumentatif, namun tetap komunikatif dan mudah dipahami oleh mahasiswa atau peneliti pemula. "
                        "Anda memiliki akses ke tools untuk mengambil konteks dari analisis sebelumnya dan untuk memeriksa data frame secara langsung. "
                        "Setiap respon harus menyertakan dasar analisis data, alasan logis, dan interpretasi yang kuat untuk penulisan ilmiah. "
                        "Gunakan Markdown untuk memformat output Anda (misalnya, tabel, judul, daftar). "
                        "Selalu prioritaskan penggunaan informasi dari dataset yang diunggah dan konteks yang diberikan."
                        
                        # --- PENYESUAIAN PENTING DI SINI ---
                        "**Pikirkan langkah demi langkah.** Ketika Anda perlu menggunakan tools, panggil **satu tool saja pada satu waktu**."
                        "Analisis output dari tool tersebut sebelum memutuskan langkah atau tool berikutnya."
                        "Jika sebuah pertanyaan membutuhkan beberapa informasi dari dataset, Anda mungkin perlu memanggil `dataframe_inspector_tool` beberapa kali secara berurutan, satu per satu, bukan sekaligus."
                        "Setelah semua informasi yang relevan terkumpul, gabungkan dan sajikan dalam respons akhir yang komprehensif."
                        # --- AKHIR PENYESUAIAN PENTING ---

                        "Jika pengguna meminta analisis data spesifik atau informasi tentang struktur data, gunakan 'dataframe_inspector_tool'."
                        "Jika pengguna mencari konteks dari percakapan atau analisis sebelumnya, gunakan 'context_retriever_tool'."
                        "Jika Anda menggunakan tool, sampaikan hasil penggunaan tool tersebut secara ringkas dan informatif, kemudian lanjutkan dengan interpretasi atau jawaban akhir."
                    )
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(st.session_state.llm, TOOLS_FOR_AGENT, prompt)
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=TOOLS_FOR_AGENT, verbose=True)

        st.session_state._last_api_key_used = google_api_key
        st.success("Model Gemini dan Agen berhasil diinisialisasi.")
        
    except Exception as e:
        st.error(f"Gagal menginisialisasi model Gemini atau agen. Pastikan API Key Anda benar dan aktif. Error: {e}")
        st.session_state.llm = None
        st.session_state.embeddings = None
        st.session_state.agent_executor = None
        st.session_state._last_api_key_used = None
        st.stop()

# Inisialisasi Text Splitter untuk RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# --- Fungsi Utility untuk Vector Store ---
def create_vector_store(text_chunks):
    """Membuat atau memperbarui FAISS vector store dari chunk teks."""
    if text_chunks and st.session_state.embeddings:
        try:
            vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state.embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Gagal membuat vector store: {e}")
            return None
    return None

def add_to_vector_store(current_vector_store, new_text_chunks):
    """Menambahkan chunk teks baru ke vector store yang sudah ada."""
    if not new_text_chunks:
        return current_vector_store
    if current_vector_store is None:
        return create_vector_store(new_text_chunks)
    else:
        if st.session_state.embeddings is None:
            st.error("Embeddings model belum diinisialisasi.")
            return current_vector_store
        try:
            current_vector_store.add_texts(new_text_chunks)
            return current_vector_store
        except Exception as e:
            st.error(f"Gagal menambahkan ke vector store: {e}")
            return current_vector_store


# --- Pemrosesan Unggahan File ---
if uploaded_file is not None and (st.session_state.dataframe is None or uploaded_file.name != st.session_state.get('uploaded_file_name')):
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataframe = df
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success("Dataset berhasil diunggah!")
        st.write("5 Baris Pertama Dataset:")
        st.dataframe(df.head())

        initial_summary_text = f"Dataset telah diunggah dengan {df.shape[0]} baris dan {df.shape[1]} kolom.\n"
        initial_summary_text += f"Nama Kolom: {df.columns.tolist()}\n"
        initial_summary_text += "Statistik Deskriptif (hanya kolom numerik):\n"
        
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce') 
            except Exception:
                pass 

        df_numeric = df.select_dtypes(include=['number']).dropna(axis=1, how='all')

        if not df_numeric.empty:
            initial_summary_text += df_numeric.describe().to_markdown()
        else:
            initial_summary_text += "Tidak ada kolom numerik yang ditemukan atau semua nilai numerik kosong setelah pembersihan."

        st.session_state.analysis_summary = initial_summary_text
        
        chunks = text_splitter.split_text(initial_summary_text)
        st.session_state.vector_store = add_to_vector_store(st.session_state.vector_store, chunks)
        
        st.session_state.chat_history.append({"role": "assistant", "content": 
            "Dataset Anda telah diunggah dan ringkasan awal telah dibuat. "
            "Anda sekarang bisa mulai bertanya tentang analisis data atau penulisan ilmiah!"})
        st.session_state.chat_history.append({"role": "assistant", "content": st.session_state.analysis_summary})

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
        st.session_state.dataframe = None
        st.session_state.uploaded_file_name = None
        st.session_state.analysis_summary = ""
        st.session_state.vector_store = None


# --- Tampilan Riwayat Chat ---
st.subheader("💬 Diskusi & Analisis")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Fungsi untuk Memproses Input Pengguna dengan Agen ---
def get_agent_response(question):
    if st.session_state.dataframe is None:
        return "Mohon unggah dataset terlebih dahulu untuk memulai analisis."

    formatted_chat_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=msg["content"]))
        else:
            formatted_chat_history.append(AIMessage(content=msg["content"]))

    try:
        response = st.session_state.agent_executor.invoke({
            "input": question,
            "chat_history": formatted_chat_history
        })
        
        final_response_content = response.get("output", "Tidak ada respons dari agen.")
        
        response_chunks = text_splitter.split_text(final_response_content)
        st.session_state.vector_store = add_to_vector_store(st.session_state.vector_store, response_chunks)

        return final_response_content
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses permintaan Anda oleh agen: {e}")
        return "Terjadi kesalahan internal saat memproses permintaan Anda. Mohon coba lagi."

# --- Input Pengguna ---
user_question = st.chat_input("Tanyakan sesuatu tentang dataset atau minta bantuan penulisan ilmiah...")

if user_question:
    if st.session_state.dataframe is not None:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis dan menyusun jawaban..."):
                response_content = get_agent_response(user_question)
                st.markdown(response_content)
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
    else:
        st.error("Mohon unggah dataset terlebih dahulu sebelum mengajukan pertanyaan.")