from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from huggingface_hub import hf_hub_download

# Import shared utils
from utils.config import REPO_ID, MODEL_FILENAME

from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.processing import preprocess_image
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.report_generator import generate_docx_report

from utils.logger import get_logger

logger =get_logger(__name__)

# Create Router
router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# Global Model Variable
MODEL = None

def load_model():
    """Loads model from Hugging Face Hub"""
    global MODEL
    if MODEL is None:
        try:
            logger.info("Loading model from Hugging Face...")
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            custom_objects = {
                "Avg2MaxPooling": Avg2MaxPooling, 
                "DepthwiseSeparableConv": DepthwiseSeparableConv
            }
            MODEL = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Helper

def get_image_base64(image):
    buffered = io.BytesIO() #BytesIO() is like a temporary file that exists in RAM.
    image.save(buffered,format="PNG")

    return base64.b64encode(buffered.getvalue())# binary data -> plain text





#----------router----------------------------------


# --------------------------------------------------
# Application Startup
# --------------------------------------------------

# Runs once when FastAPI starts.
# Loads the ML model into memory so it is available
# for all incoming requests.

@router.on_event("startup")
def startup_event():
    load_model()


# --------------------------------------------------
# Home Page
# --------------------------------------------------

# Returns the application's home page.
#
# No file I/O, database call, network call, or await.
# Could be synchronous.
@router.get("/", response_class=HTMLResponse)
def read_root(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("frontend/static/favicon.ico")

# **Short Description:**

# * `request: Request` receives the incoming HTTP request from FastAPI.
# * `TemplateResponse()` renders the `index.html` page.
# * `{"request": request}` passes the request object to the Jinja2 template so it can access request-related information if needed (e.g., `{{ request.url }}`).
# * When a user visits `/`, the server renders and returns the HTML page.




#---------------------------------------------------------------------------------------------

# --------------------------------------------------
# Analyze Image
# --------------------------------------------------

#---------------------------------------------------------------------------------------------

# Async is required because:
#     await file.read()
#
# Reading uploaded files is an I/O operation.
# While waiting for file data, FastAPI can process
# other requests instead of blocking.

#---------------------------------------------------------------------------------------------
# file: UploadFile receives the uploaded file from the client.
# File(...) tells FastAPI that this parameter is required and should come from a file upload (multipart/form-data).
# UploadFile provides useful attributes and methods such as:
# file.filename → uploaded file name
# file.content_type → file MIME type
# await file.read() → read file contents
# async is used because the function performs asynchronous file reading with await file.read().
#---------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
# multipart/form-data is used to send files and form data (such as images, PDFs, and documents) from the client to the server in an HTTP request.
# UploadFile provides useful attributes and methods such as:
# file.filename → uploaded file name
# file.content_type → file MIME type
# await file.read() → read file contents
#--------------------------------------------------------------------------------------
@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    logger.info(
        f"Analyze request received for file: {file.filename}"
    )

    # Ensure model is available.
    if MODEL is None:

        load_model()

        if MODEL is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not loaded"}
            )

    # --------------------------------------------------
    # Read Uploaded File
    # --------------------------------------------------

    # Asynchronous file read.
    # FastAPI releases control to the event loop while
    # waiting for uploaded file data.
    #The data is loaded as bytes
    contents = await file.read()

    # Convert raw bytes into an in-memory file object.
    image_stream = io.BytesIO(contents)

    # Open image using Pillow.
    image = Image.open(image_stream)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    # Resize, normalize, convert to numpy array, etc.
    processed_img = preprocess_image(image)

    # --------------------------------------------------
    # Model Prediction
    # --------------------------------------------------

    # TensorFlow/Keras inference.
    # This is CPU-bound and synchronous.
    preds = MODEL.predict(processed_img)

    score = float(preds[0][0])

    is_malignant = score > 0.5

    # --------------------------------------------------
    # Grad-CAM Generation
    # --------------------------------------------------

    gradcam_b64 = None

    try:

        # Search from last layer backward
        # to find the final convolution layer.
        last_conv = next(
            (layer.name for layer in MODEL.layers[::-1]
                if "depthwise_separable_conv" in layer.name
            ),
            None
        )

        if last_conv:

            heatmap = make_gradcam_heatmap(
                processed_img,
                MODEL,
                last_conv
            )

            if heatmap is not None:

                gradcam_img = save_and_display_gradcam(
                    image,
                    heatmap
                )

                # Convert image into Base64 string
                # for frontend display.
                gradcam_b64 = get_image_base64(
                    gradcam_img
                )

    except Exception as e:

        logger.warning(
            f"Grad-CAM generation failed: {e}"
        )

    # --------------------------------------------------
    # JSON Response
    # --------------------------------------------------

    return {
        "label":
            "Malignant (Cancerous)"
            if is_malignant
            else "Benign (Non-Cancerous)",

        "score": score,

        "percent":
            score * 100
            if is_malignant
            else (1 - score) * 100,

        "class_id":
            1 if is_malignant else 0,

        "is_malignant":
            is_malignant,

        "gradcam_image":
            gradcam_b64
    }


# --------------------------------------------------
# Generate Report
# --------------------------------------------------

# Async is required because:
#     await file.read()
#
# File upload reading is I/O-bound.
# FastAPI can serve other requests while waiting.
@router.post("/report")
async def get_report(file: UploadFile = File(...)):

    logger.info(
        f"Report request received for file: {file.filename}"
    )

    try:

        # --------------------------------------------------
        # Ensure Model Exists
        # --------------------------------------------------

        if MODEL is None:

            load_model()

            if MODEL is None:

                return JSONResponse(
                    status_code=503,
                    content={"error": "Model not loaded"}
                )

        # --------------------------------------------------
        # Read Uploaded File
        # --------------------------------------------------

        # Async file read.
        # Event loop can handle other requests while
        # waiting for upload data.
        contents = await file.read() #coms in bytes form 

        image = Image.open(
            io.BytesIO(contents)
        )


        # --------------------------------------------------
        # Re-run Prediction
        # --------------------------------------------------

        processed_img = preprocess_image(image)

        preds = MODEL.predict(processed_img)

        score = float(preds[0][0])

        is_malignant = score > 0.5

        label = (
            "Malignant (Cancerous)"
            if is_malignant
            else "Benign (Non-Cancerous)"
        )

        conf_percent = (
            score * 100
            if is_malignant
            else (1 - score) * 100
        )

        # --------------------------------------------------
        # Re-run Grad-CAM
        # --------------------------------------------------

        gradcam_bytes = None

        try:

            last_conv = next(
                (
                    layer.name
                    for layer in MODEL.layers[::-1]
                    if "depthwise_separable_conv"
                    in layer.name
                ),
                None
            )

            if last_conv:

                heatmap = make_gradcam_heatmap(
                    processed_img,
                    MODEL,
                    last_conv
                )

                if heatmap is not None:

                    gradcam_img = save_and_display_gradcam(
                        image,
                        heatmap
                    )

                    # In-memory file object.
                    gradcam_bytes = io.BytesIO()

                    # Save image into memory.
                    gradcam_img.save(
                        gradcam_bytes,
                        format="PNG"
                    )

                    # Move cursor to beginning.
                    gradcam_bytes.seek(0)

        except Exception:
            pass

        # --------------------------------------------------
        # Original Image Buffer
        # --------------------------------------------------

        img_bytes = io.BytesIO()

        image.save(
            img_bytes,
            format="PNG"
        )

        img_bytes.seek(0)

        # --------------------------------------------------
        # Create DOCX Report
        # --------------------------------------------------

        report_buffer = generate_docx_report(
            image_buffer=img_bytes,
            prediction_label=label,
            confidence_score=score,
            confidence_percent=conf_percent,
            gradcam_buffer=gradcam_bytes
        )

        report_buffer.seek(0)

        # --------------------------------------------------
        # Return Downloadable File
        # --------------------------------------------------

        headers = {
            "Content-Disposition":
            'attachment; filename="thyroid_analysis_report.docx"'
        }

        return StreamingResponse(
            report_buffer,
            headers=headers,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
        )

    except Exception as e:

        logger.error(
            f"Report generation error: {e}"
        )

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )