# adapted from https://github.com/jojo96/ASCIIGan/blob/main/asciiGan.py
import ray
import requests

@ray.remote
def send_query(data):
    resp = requests.post("http://localhost:8000/", data=data)
    return resp.json()

import time

###############################################################
import streamlit as st
from PIL import Image, ImageDraw

st.header("Model Serving Demo")
st.write("Choose any image to run inference on")


uploaded_files = st.file_uploader("Choose an image(s)...",type=['jpg','jpeg','png'] , accept_multiple_files=True)


def run_single_inference(image_bytes: bytes):
    resp = requests.post("http://localhost:8000/", data=image_bytes)
    return resp.json()

def run_parallel_inference(image_bytes_list: list):
    futures = []
    for image_bytes in image_bytes_list:
        futures.append(send_query.remote(image_bytes))
    return ray.get(futures)


col1, col2= st.columns(2)

if n_files := len(uploaded_files):
    print(f'n_files {n_files}')
    if n_files == 1:
        uploaded_file = uploaded_files[0]

        bytes_data = uploaded_file.getvalue()
        start = time.time()
        prediction = run_single_inference(bytes_data)
        t_prediction = time.time() - start

        boxes = prediction['pred_boxes']
        classes = prediction['pred_classes']
        labels = classes

        output = Image.open(uploaded_file)
        draw = ImageDraw.Draw(output)

        with col1:
            st.image(uploaded_file, caption="Input Image", use_column_width=True)
        for box in boxes:
            draw.rectangle(box, outline=(0,255,0), width = 1)
        with col2:
            st.image(output, caption="Prediction", use_column_width=True)

        elapsed = time.time() - start
        st.text(f'[INFO] total elapsed time: {elapsed}. API call took: {t_prediction}.')

    else:
        bytes_data_list = [uploaded_file.getvalue() for uploaded_file in uploaded_files]
        start = time.time()
        predictions = run_parallel_inference(bytes_data_list)
        t_prediction = time.time() - start

        for i, uploaded_file in enumerate(uploaded_files):
            prediction = predictions[i]

            boxes = prediction['pred_boxes']
            classes = prediction['pred_classes']
            labels = classes

            output = Image.open(uploaded_file)
            draw = ImageDraw.Draw(output)

            with col1:
                st.image(uploaded_file, caption="Input Image", use_column_width=True)
            for box in boxes:
                draw.rectangle(box, outline=(0,255,0), width = 1)
            with col2:
                st.image(output, caption="Prediction", use_column_width=True)

        elapsed = time.time() - start
        st.text(f'[INFO] total elapsed time: {elapsed}. API call took: {t_prediction}.')