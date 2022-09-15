# adapted from https://github.com/jojo96/ASCIIGan/blob/main/asciiGan.py
from io import BytesIO

import ray
import requests

BATCH_SIZE = 1
@ray.remote
def send_query(data):
    resp = requests.post("http://localhost:8000/", data=data)
    return resp.json()

###############################################################
import streamlit as st

st.header("Model Serving Demo")
st.write("Choose any image to run inference on")

uploaded_file = st.file_uploader("Choose an image...")


def run_inference(image_bytes:BytesIO):

    # # test code
    # import cv2
    # import numpy as np
    # image_np = np.frombuffer(image_bytes, np.uint8)
    # original_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # print(original_image)

    return ray.get([send_query.remote(image_bytes) for i in range(BATCH_SIZE)])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Image", use_column_width=True)

    bytes_data = uploaded_file.getvalue()
    inference_results = run_inference(bytes_data)
    for inference_result in inference_results:
        st.text(inference_result)
    # st.image(image, caption="Inference Result", use_column_width=True)
