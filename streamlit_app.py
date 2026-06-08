from fileinput import filename
import os
from sys import exception
import warnings

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from seaborn import heatmap
import streamlit as st
import tensorflow as tf
from PIL import Image
import io
import numpy as np
from huggingface_hub import hf_hub_download

from utils.config import REPO_ID, MODEL_FILENAME
from utils.processing import preprocess_image
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.report_generator import generate_docx_report
from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.logger import get_logger



logger = get_logger(__name__)

# --- page Config - - 

st.set_page_config(
    page_title="ThyroCheck AI | Cancer Detection System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered Thyroid Cancer Detection & Diagnostic Reporting System"
    }
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.5rem; }
    </style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    try:
        logger.info("Loding model for streamlit...")
        path = hf_hub_download(repo_id=REPO_ID,filename=MODEL_FILENAME)
        custom_object =  {"Avg2MaxPooling": Avg2MaxPooling, "DepthwiseSeparableConv": DepthwiseSeparableConv}
        model = tf.keras.models.load_model(path,custom_objects=custom_object,compile=False)
        logger.info('Streamlit model loaded successfully')
        return model

    except Exception as e:
        logger.error(f"Streamlit model failed to load: {e}")
        return None

def main ():
    st.title("Thyroid cancer detection system")

    st.write('Upload a thyroid medical image(ultrasound / pathology) for AI=powered cancer detection')
    model = load_model()

    if not model:
        st.error("Model failed to load . Check configuration/ internet.")
        st.stop()

    else: 
        st.toast("✅ Model loaded successfully!")
        pass

    uploaded_file = st.file_uploader('Choose a thyroid image ', type = ['png','jpg','jpeg'])
    
    if uploaded_file:
        logger.info(f'file uploaded in streamlit : {uploaded_file.name}')
        image= Image.open(uploaded_file)
        st.image(image,caption='Uploaded image',width='stretch')

        with st.spinner("Analyzing"):
            #Prediction
            processed_img = preprocess_image(image)
            preds = model.predict(processed_img)
            
        # Probability of Class 1 (Malignant/Cancerous) returned by sigmoid output
            score = float(preds[0][0])

            is_cancer = score >0.5
            label = "Malignant (Cancerous)" if is_cancer else "Benign (Non-Cancerous)"
            conf_percent = score * 100 if is_cancer else (1-score) *100

            # Display Results
            st.markdown("----")
            st.markdown("### 🔬 Analysis Results")
            c1,c2 = st.columns(2)
            c1.metric("Prediction",label)
            c1.metric("Class", "1" if is_cancer else "0")
            c2.metric("Confidence Score of having cancer", f"{score:.4f}")
            c2.metric(f"Confidence of(   {label}   ) class %", f"{conf_percent:.2f}%")

            #2. Grad-CAM:
            st.markdown("---")
            st.markdown("### 🧠 Interpretability (Grad-CAM)")

            gradcam_img= None
            try:
                # Find last depthwise conv layer dynamically
                last_conv = next((l.name for l in model.layers[::-1] if "depthwise_separable_conv" in l.name),None)

                if last_conv:
                    heatmap = make_gradcam_heatmap(processed_img, model, last_conv)
                    if heatmap is not None:
                        gradcam_img = save_and_display_gradcam(image, heatmap)
                        gc1, gc2 = st.columns(2)
                        gc1.image(image, caption="Original", width="stretch")
                        gc2.image(gradcam_img, caption="Grad-CAM Heatmap", width="stretch")
                else:
                    st.warning("Layer for Grad-CAM not found.")
            except Exception as e:
                st.error(f"Grad-CAM Error: {e}")


                    # 3. Report Generation
            st.markdown("---")
            st.markdown("### 📄 Download Report")
            
            if gradcam_img: # Only generate report if analysis complete
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                grad_bytes = io.BytesIO()
                gradcam_img.save(grad_bytes, format='PNG')
                grad_bytes.seek(0)
                
                report = generate_docx_report(img_bytes, label, score, conf_percent, grad_bytes)
                
                st.download_button(
                    label="Download Report (DOCX)",
                    data=report,
                    file_name="thyroid_analysis_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )



with st.sidebar:
    st.markdown("### 👨‍💻 Developer shivansh")
    st.markdown("[🌍 Portfolio](https://my-personal-portfolio-ten-pearl.vercel.app/)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/shivanshvyas/)")
    st.markdown("[🐙 GitHub](https://github.com/Shivanshvyas1729)")
if __name__ == "__main__":
    main()
