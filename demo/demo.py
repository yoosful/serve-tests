# adapted from https://github.com/jojo96/ASCIIGan/blob/main/asciiGan.py
import ray
import requests

BATCH_SIZE = 1
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

uploaded_file = st.file_uploader("Choose an image...")


def run_inference(image_bytes: bytes):

    # # test code
    # import cv2
    # import numpy as np
    # image_np = np.frombuffer(image_bytes, np.uint8)
    # original_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # print(original_image)

    return ray.get([send_query.remote(image_bytes) for i in range(BATCH_SIZE)])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Image", use_column_width=True)

    start = time.time()
    bytes_data = uploaded_file.getvalue()
    predictions = run_inference(bytes_data)
    for prediction in predictions:
        st.text(prediction)

        # # test code
        # prediction = {
        #     'pred_boxes': [
        #         [126.60348510742188, 244.89768981933594, 459.8291015625, 480.0], [251.10829162597656, 157.8126983642578, 338.97314453125, 413.6379089355469], [114.84960174560547, 268.6864318847656, 148.2351531982422, 398.81109619140625], [0.8217036724090576, 281.03265380859375, 78.60723876953125, 478.42095947265625], [49.39535140991211, 274.12286376953125, 80.154541015625, 342.98077392578125], [561.2247924804688, 271.5816345214844, 596.2755126953125, 385.2552185058594], [385.9071960449219, 270.3125, 413.7130432128906, 304.0396728515625], [515.9295043945312, 278.3743591308594, 562.2792358398438, 389.3802490234375], [335.24090576171875, 251.9166717529297, 414.74908447265625, 275.9375305175781], [350.9300231933594, 269.2059631347656, 386.09844970703125, 297.9080505371094], [331.6292419433594, 230.99961853027344, 393.27593994140625, 257.2008972167969], [510.73486328125, 263.26556396484375, 570.9864501953125, 295.9194030761719], [409.0841064453125, 271.86456298828125, 460.5582275390625, 356.8721618652344], [506.8766784667969, 283.32574462890625, 529.9403076171875, 324.0391845703125], [594.5662841796875, 283.48199462890625, 609.0577392578125, 311.412353515625]
        #     ],
        #     'pred_classes': [17, 0, 0, 0, 0, 0, 0, 0, 25, 0, 25, 25, 0, 0, 24],
        # }

        boxes = prediction['pred_boxes']
        classes = prediction['pred_classes']
        labels = classes

        output = Image.open(uploaded_file)
        draw = ImageDraw.Draw(output)
        for box in boxes:
            draw.rectangle(box, outline=(0,255,0), width = 1)

        st.image(output, caption="Prediction", use_column_width=True)

    elapsed = time.time() - start
    st.info(f'[INFO] total elapsed time: {elapsed}')
