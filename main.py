import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
from PIL import Image



json_file = open('agri_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("agri_model.h5")
# print("Loaded model from disk")

st.title('Identify Animals By Their Images')
class_name = ['Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)_Common_rust', 'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy', 'Cotton_leaf_diseased', 'Cotton_leaf_fresh', 'Orange_Haunglongbing_(Citrus_greening)', 'Orange__Black_Rot', 'Orange__Canker', 'Orange__Healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Random', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy', 'Wheat_healthy', 'Wheat_leaf_rust', 'Wheat_nitrogen_deficiency']

traslate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
            "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel"}


genre = st.radio(
    "How You Want To Upload Your Image",
    ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file")

# ImagePath = st.file_uploader("Choose a file")

if ImagePath is not None:

    try:
        image_ = Image.open(ImagePath)

        st.image(image_, width=250)

    except UnidentifiedImageError:
        st.write('Input Valid File Format !!!  [ jpeg, jpg, png only this format is supported ! ]')


try:
    if st.button('Predict'):
        test_image = image.load_img(ImagePath, target_size=(256, 256))
        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        result = loaded_model.predict(test_image, verbose=0)

        # type_ = traslate[class_name[np.argmax(result)]]
        type_ = class_name[np.argmax(result)]

        st.header('Prediction is: ' + type_)
        st.header('Confidence is: ' + str(round(np.max(result), 4) * 100) + ' %')

except TypeError:
    st.header('Please Upload Your File !!!')

except UnidentifiedImageError:
    st.header('Input Valid File !!!')
