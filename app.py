import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("simple_cnn_model.h5")

st.title("🎨 명화 vs AI 그림 분류기")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="업로드한 이미지", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        st.subheader("결과: 🎨 AI 그림입니다!")
    else:
        st.subheader("결과: 🖼️ 진짜 명화입니다!")
