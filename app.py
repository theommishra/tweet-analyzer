import streamlit as st
import pandas as pd
from model import EmotionClassifier, EKMAN_EMOTIONS, EMOTION_EMOJI

# Page config
st.set_page_config(
    page_title="TWITTER EMOTION ANALYZER",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading using @st.cache_resource
@st.cache_resource(show_spinner="Loading Emotion Classifier Engine (SentenceTransformer + LR)...")
def get_classifier():
    return EmotionClassifier()

# Inject custom monochrome / black-and-white CSS
st.markdown("""
<style>
    /* Dark monochrome base styling */
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Card Container */
    .mono-card {
        background-color: #121212;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .mono-header {
        border-bottom: 1px solid #333333;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }

    /* Grayscale metrics and text */
    .mono-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #FFFFFF;
        text-transform: uppercase;
        margin: 0;
    }

    .mono-subtitle {
        font-size: 0.9rem;
        color: #888888;
        letter-spacing: 1px;
        margin-top: 4px;
        text-transform: uppercase;
    }

    .emotion-badge {
        display: inline-block;
        background-color: #1F1F1F;
        border: 1px solid #444444;
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Input area customization */
    .stTextArea textarea {
        background-color: #0B0B0B !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #666666 !important;
        box-shadow: none !important;
    }

    /* Button styling */
    .stButton button {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #333333 !important;
        border-color: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* Force all progress bars to be grayscale */
    .stProgress > div > div > div > div {
        background-color: #FFFFFF !important;
    }
    .stProgress > div > div {
        background-color: #222222 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #121212 !important;
        border: 1px solid #333333 !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
    }
    
    /* Code/JSON block monochrome */
    code, pre {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        color: #DDDDDD !important;
    }
</style>
""", unsafe_allow_html=True)

# Load classifier
classifier = get_classifier()

# Sidebar - Model Meta & Info
with st.sidebar:
    st.markdown('<div class="mono-title" style="font-size: 1.2rem;">SYSTEM INFO</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Architecture:**  \n`SentenceTransformer + LR + NRC`")
    st.markdown(f"**Model Name:**  \n`{getattr(classifier, 'embedding_model_name', 'all-MiniLM-L6-v2')}`")
    st.markdown(f"**Version:**  \n`{getattr(classifier, 'model_version', 'v4')}`")
    f1_val = getattr(classifier, "cv_macro_f1", None)
    f1_str = f"{f1_val * 100:.1f}%" if f1_val else "N/A"
    st.markdown(f"**CV Macro-F1:** `{f1_str}`")
    st.markdown("---")
    st.markdown("**Taxonomy:** 7 Ekman Classes")
    for e in EKMAN_EMOTIONS:
        st.markdown(f"- `{e.upper()}`")

# Header Section
st.markdown("""
<div class="mono-header">
    <div class="mono-title">TWITTER EMOTION ANALYZER</div>
    <div class="mono-subtitle">Offline Ekman-7 Emotion Detection for Twitter Posts</div>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("##### ENTER TWEET OR TWITTER POST TO ANALYZE")

# Preset selector buttons for testing
sample_sentences = {
    "Joy": "Just got the best news ever!! So happy and grateful rn 🎉 #blessed #joy",
    "Sadness": "Missing my best friend every single day... the emptiness never goes away 😢 #heartbroken",
    "Anger": "Stop lying to us!! This is absolutely outrageous and unacceptable 😡 #furious #angry",
    "Fear": "The situation is getting worse and I'm genuinely scared of what comes next 😱 #worried",
    "Surprise": "WAIT WHAT?! I cannot believe this just happened omg 😲 #shocked #mindblown",
    "Disgust": "This is absolutely disgusting behavior from a public figure. Absolutely vile 🤢 #unacceptable",
    "Neutral": "The quarterly earnings report will be released at 4:00 PM EST on Friday."
}

preset_cols = st.columns(len(sample_sentences))
selected_preset = None
for idx, (label, sample_text) in enumerate(sample_sentences.items()):
    if preset_cols[idx].button(label, key=f"btn_{label}"):
        selected_preset = sample_text

default_text = selected_preset if selected_preset else ""

input_text = st.text_area(
    label="Tweet Input",
    value=default_text,
    height=130,
    placeholder="Paste a tweet here (e.g. 'Just got amazing news, feeling absolutely thrilled! 🎉 #blessed')...",
    label_visibility="collapsed"
)

col_btn, col_blank = st.columns([1, 4])
with col_btn:
    analyze_clicked = st.button("ANALYZE TWEET", use_container_width=True)

# Process & Display Results
if analyze_clicked or input_text.strip():
    if not input_text.strip():
        st.warning("Please paste or enter a tweet to analyze.")
    else:
        with st.spinner("Analyzing tweet..."):
            res = classifier.predict(input_text)
        
        primary = res.get("primary_emotion", "neutral").upper()
        confidence_pct = res.get("confidence_percentage", 0.0)
        emoji = EMOTION_EMOJI.get(primary.lower(), "⚫")
        breakdown = res.get("breakdown", [])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid layout for primary result & taxonomy distribution
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"""
            <div class="mono-card">
                <div style="font-size: 0.85rem; color: #888888; letter-spacing: 1px; margin-bottom: 12px;">PRIMARY DETECTED EMOTION</div>
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;">
                    <span style="font-size: 2.8rem;">{emoji}</span>
                    <span class="emotion-badge">{primary}</span>
                </div>
                <div style="border-top: 1px solid #222222; padding-top: 16px; margin-top: 16px;">
                    <div style="font-size: 0.8rem; color: #888888; letter-spacing: 1px;">CONFIDENCE RATING</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #FFFFFF;">{confidence_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("""
            <div class="mono-card">
                <div style="font-size: 0.85rem; color: #888888; letter-spacing: 1px; margin-bottom: 16px;">TAXONOMY DISTRIBUTION (EKMAN-7)</div>
            """, unsafe_allow_html=True)
            
            for item in breakdown:
                emo_name = item["emotion"].upper()
                score_pct = item["percentage"]
                
                b_col1, b_col2, b_col3 = st.columns([3, 6, 2])
                with b_col1:
                    st.markdown(f"<span style='font-size: 0.85rem; font-weight: 600; letter-spacing: 1px;'>{emo_name}</span>", unsafe_allow_html=True)
                with b_col2:
                    st.progress(score_pct / 100.0)
                with b_col3:
                    st.markdown(f"<span style='font-size: 0.85rem; color: #AAAAAA; font-family: monospace;'>{score_pct:5.1f}%</span>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Collapsible Preprocessing & Feature Breakdown
        with st.expander("PREPROCESSING TRACE & FEATURE BREAKDOWN"):
            prep_details = res.get("preprocessing_details", {})
            cleaned = res.get("cleaned_text", "")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**Original Text:**")
                st.code(input_text, language="text")
                st.markdown("**Cleaned & Normalized Text:**")
                st.code(cleaned, language="text")
            
            with p_col2:
                st.markdown("**Extracted Tokens:**")
                tokens = prep_details.get("tokens", [])
                st.code(", ".join(tokens) if tokens else "None", language="text")
                st.markdown("**Text Statistics:**")
                st.json({
                    "raw_length": prep_details.get("raw_length", len(input_text)),
                    "cleaned_length": prep_details.get("cleaned_length", len(cleaned)),
                    "token_count": prep_details.get("token_count", len(tokens)),
                    "has_emoji": prep_details.get("has_emoji", False),
                    "has_exclamation": prep_details.get("has_exclamation", False),
                    "has_question": prep_details.get("has_question", False),
                })
