import numpy as np

from utils.logger import logger
from PIL import Image
def preprocessing_image(image):
    """
    Preprocesses the image for the FibonacciNet model.
    Steps:
    1. Resize to target size (224, 224).
    2. Convert to numpy array.
    3. Ensure 3 channels (RGB).
    4. batch dimension expansion.
    5. Rescale pixel values to [0, 1] (Standard for custom trained models).
    """

    logger.info(f"Preprocessing image of size : {image.size} and mode : {image.mode}")

    if image.mode != "RGB":
        image= image.convert("RGB")

    
    image = image.resize((224,224))

    img_array= np.array(image)

    img_array= np.expand_dims(img_array,axis=0)

    img_array = img_array.astype("float32") /255.0  #(batch_size, height, width, channels)
    
    return img_array