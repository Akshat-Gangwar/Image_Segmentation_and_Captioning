import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from model import get_caption_model, generate_caption
import tensorflow as tf
import io
import requests
import cv2
import os
@st.cache_resource
def get_model():
    return get_caption_model()

caption_model = get_model()

@st.cache_resource
def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)
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

def draw_combined_output(image, outputs, score_threshold=0.5):
    image_np = np.array(image).copy()
    boxes = outputs['boxes']
    labels = outputs['labels']
    scores = outputs['scores']
    masks = outputs['masks']

    for i in range(len(masks)):
        if scores[i] < score_threshold:
            continue

        mask = masks[i, 0].detach().cpu().numpy()
        mask_binary = mask > 0.5
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        image_np[mask_binary] = image_np[mask_binary] * 0.5 + color * 0.5

        
        box = boxes[i].detach().cpu().numpy().astype(int)
        label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
        score = scores[i].item()
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
        cv2.putText(image_np, f"{label} ({score:.2f})", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, color.tolist(), 3)

    return Image.fromarray(image_np.astype(np.uint8))


def get_binary_masks(outputs, score_threshold=0.5):
    masks = outputs['masks']
    labels = outputs['labels']
    scores = outputs['scores']

    mask_images = []
    for i in range(len(masks)):
        if scores[i] < score_threshold:
            continue
        mask = masks[i, 0].detach().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
        mask_pil = Image.fromarray(binary_mask).convert("L")
        mask_images.append((label, scores[i].item(), mask_pil))
    return mask_images

st.title("🧠 MS COCO Image Segmentation")
st.write("Upload an image and see image captions and object segmentation masks!")


img_url = st.text_input(label='Enter Image URL')
if (img_url != "") and (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB')
    st.image(img, caption="Original Image", use_container_width =True)
    img.save('tmp.jpg')
    with st.spinner("Running caption generation..."):
        predict()
    with st.spinner("Running segmentation..."):
        input_tensor = preprocess_image(img)
        with torch.no_grad():
            output = model(input_tensor)[0]

        result_img = draw_combined_output(img, output)
        mask_only_images = get_binary_masks(output)

    st.subheader("🔍 Segmented Image with Overlays")
    st.image(result_img, caption="Segmented Image", use_container_width =True)
    st.subheader("🖼️ Individual Masks")
    if mask_only_images:
        for idx, (label, score, mask_img) in enumerate(mask_only_images):
            st.markdown(f"**{label}** – Confidence: {score:.2f}")
            st.image(mask_img, use_container_width =True)
    else:
        st.info("No masks detected above threshold.")
    os.remove('tmp.jpg')

st.markdown('<center style="opacity: 70%">OR</center>', unsafe_allow_html=True)
img_upload = st.file_uploader(label='Upload Image', type=['jpg', 'png', 'jpeg'])

if img_upload != None:
    img = img_upload.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img.save('tmp.jpg')
    st.image(img)
    with st.spinner("Running caption generation..."):
        predict()
    with st.spinner("Running segmentation..."):
        input_tensor = preprocess_image(img)
        with torch.no_grad():
            output = model(input_tensor)[0]

        result_img = draw_combined_output(img, output)
        mask_only_images = get_binary_masks(output)

    st.subheader("🔍 Segmented Image with Overlays")
    st.image(result_img, caption="Segmented Image", use_container_width =True)
    st.subheader("🖼️ Individual Masks")
    if mask_only_images:
        for idx, (label, score, mask_img) in enumerate(mask_only_images):
            st.markdown(f"**{label}** – Confidence: {score:.2f}")
            st.image(mask_img, use_container_width =True)
    else:
        st.info("No masks detected above threshold.")
    os.remove('tmp.jpg')
