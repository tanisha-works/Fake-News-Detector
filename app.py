import streamlit as st
import pickle
import re
import string
import pandas as pd
from deep_translator import GoogleTranslator

# 1. Load Model & Vectorizer
model_lr = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Config & Styling ---
st.set_page_config(page_title="Veritas AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    /* Navbar styling */
    .nav { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background-color: #111827; color: white; border-radius: 10px; margin-bottom: 20px;}
    .nav a { color: white; text-decoration: none; font-size: 24px; }
    
    /* Main container */
    .main { background-color: #f8fafc; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #2563eb; color: white; font-weight: bold; border: none;}
    .stTextArea>div>div>textarea { border-radius: 10px; border: 1px solid #cbd5e1; }
    
    /* Sidebar Project Card */
    .project-card { background-color: #f1f5f9; padding: 15px; border-radius: 10px; border-left: 5px solid #2563eb; margin-bottom: 20px; color: #1e293b;}
    
    /* Footer */
    .footer { text-align: center; padding: 40px; color: #64748b; font-size: 14px; border-top: 1px solid #e2e8f0; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- Top Navbar ---
st.markdown("""
    <div class="nav">
        <div style="font-size: 20px; font-weight: bold;">🛡️ Veritas AI News Detector</div>
        <a href="https://github.com/YOUR_USERNAME/YOUR_REPO" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" style="filter: invert(1);">
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar (Project Details & Graph) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    st.title("Project Overview")
    
    st.markdown("""
    <div class="project-card">
    <strong>Veritas AI</strong> is a machine learning tool that identifies misinformation patterns in Hindi and English news using NLP.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Algorithm Comparison")
    # Mock data based on your project history (Logistic Regression, Decision Tree, Random Forest)
    chart_data = pd.DataFrame({
        'Algorithm': ['LR', 'DT', 'RF'],
        'Accuracy %': [98.5, 96.2, 98.1]
    })
    st.bar_chart(chart_data.set_index('Algorithm'))
    
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092128.png", caption="AI Content Analysis", use_container_width=True)

# --- Main UI ---
col_main, col_settings = st.columns([2, 0.8])

with col_main:
    st.title("Analyze News Authenticity")
    news_input = st.text_area("📰 News Content", height=300, placeholder="यहाँ खबर पेस्ट करें...")

with col_settings:
    st.markdown("### ⚙️ Settings")
    algo = st.selectbox("Select Model Engine", ["Logistic Regression", "Decision Tree", "Random Forest"])
    st.write("---")
    st.write("**How to verify:**")
    st.write("1. Input text on the left.")
    st.write("2. Ensure model is selected.")
    st.write("3. Click 'Analyze' below.")
    
    if st.button("🚀 ANALYZE NEWS"):
        if news_input:
            with st.spinner('Verifying patterns...'):
                try:
                    translated = GoogleTranslator(source='auto', target='en').translate(news_input)
                except:
                    translated = news_input
                
                cleaned = clean_text(translated)
                vectorized = vectorizer.transform([cleaned])
                prediction = model_lr.predict(vectorized)
                
                st.markdown("---")
                if prediction[0] == 0:
                    st.balloons()
                    st.success("### ✅ Result: REAL NEWS")
                else:
                    st.error("### 🚨 Result: FAKE NEWS")
        else:
            st.warning("Input required!")

# --- Footer ---
st.markdown(f"""
    <div class="footer">
        <p>Built with ❤️ by <b>Tanisha</b></p>
        <p>NLP • Scikit-Learn • Streamlit • Python</p>
    </div>
    """, unsafe_allow_html=True)
