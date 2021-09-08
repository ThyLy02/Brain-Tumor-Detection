from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import PIL.Image
import os
import json
from PIL import Image
import numpy as np

# predict

def names(number):
    if number == 1:
        return 'No, it is not a Tumor'
    else:
        return 'It is a tumor'

# main

def main():
    st.title("Tool to predict tumor and no tumor brain image")

    image_uploaded = st.sidebar.file_uploader("Choose a brain image... ", type=['png', 'jpg', 'jpeg'],
                                              accept_multiple_files=False)

    if image_uploaded is not None:
        image = PIL.Image.open(image_uploaded).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        (W, H) = image.size
        st.write("Size of image: Weight is {} - Height is {}".format(W, H))
        st.write("")

    col1, col2 = st.beta_columns(2)
    with col1:
        pred_button = st.button("Predict")

    with col2:
        if pred_button:
            predicted = names(image_uploaded)
            st.write('Output:  ', predicted)


if __name__ == '__main__':
    main()
