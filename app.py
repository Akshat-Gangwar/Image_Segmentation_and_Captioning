import io
import os
import streamlit as st
import requests
from PIL import Image
from model import get_caption_model, generate_caption
import numpy as np
import tensorflow as tf

@st.cache_resource
def get_model():
    return get_caption_model()

caption_model = get_model()
MODEL_PATH = "saved_models/segmentation_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


def predict():
    captions = []
    pred_caption = generate_caption('tmp.jpg', caption_model)

    st.markdown('#### Predicted Captions:')
    captions.append(pred_caption)

    for _ in range(4):
        pred_caption = generate_caption('tmp.jpg', caption_model, add_noise=True)
        if pred_caption not in captions:
            captions.append(pred_caption)
    
    for c in captions:
        st.write(c)

def preprocess_image(uploaded_file):
    """Read and preprocess the uploaded image"""
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def postprocess_mask(pred_mask):
    """Post-process model output to generate displayable mask"""
    pred_mask = pred_mask[0] 
    if pred_mask.shape[-1] > 1:
        pred_mask = np.argmax(pred_mask, axis=-1)
    else:
        pred_mask = pred_mask.squeeze()
    return pred_mask

def overlay_mask(image, mask, alpha=0.5):
    """Overlay mask on image for visualization"""
    color_mask = np.zeros_like(np.array(image))
    color_mask[:, :, 0] = (mask * 255).astype(np.uint8)
    overlay = Image.blend(image, Image.fromarray(color_mask), alpha=alpha)
    return overlay

st.title('Image Captioner')
img_url = st.text_input(label='Enter Image URL')

if (img_url != "") and (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB')
    st.image(img)
    img.save('tmp.jpg')
    predict()    
    img_array, img_pil = preprocess_image('tmp.jpg')
    pred_mask = model.predict(img_array)
    mask = postprocess_mask(pred_mask)
    st.subheader("Predicted Segmentation Mask")
    st.image(mask, caption="Predicted Mask", use_container_width =True, clamp=True)
    overlay = overlay_mask(img_pil, mask)
    st.subheader("Overlayed Mask on Image")
    st.image(overlay, caption="Overlay", use_container_width=True)
    os.remove('tmp.jpg')


st.markdown('<center style="opacity: 70%">OR</center>', unsafe_allow_html=True)
img_upload = st.file_uploader(label='Upload Image', type=['jpg', 'png', 'jpeg'])

if img_upload != None:
    img = img_upload.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img.save('tmp.jpg')
    st.image(img)
    predict()
    img_array, img_pil = preprocess_image('tmp.jpg')
    pred_mask = model.predict(img_array)
    mask = postprocess_mask(pred_mask)
    st.subheader("Predicted Segmentation Mask")
    st.image(mask, caption="Predicted Mask", use_container_width =True, clamp=True)
    overlay = overlay_mask(img_pil, mask)
    st.subheader("Overlayed Mask on Image")
    st.image(overlay, caption="Overlay", use_container_width =True)
    os.remove('tmp.jpg')
