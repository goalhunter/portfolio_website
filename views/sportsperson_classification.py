import streamlit as st
import cv2
import numpy as np


st.title("Sports Legends Image Classification")
st.subheader("Upload Image of Iconic Athletes from the below list: ")
st.markdown("**Lionel Messi, Virat Kohli, Roger Federer, Maria Sharapova, Serena Williams**")

import streamlit as st
import joblib
import json
import numpy as np
import base64
import cv2
import pywt

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def classify_image(image_base64_data, file_path=None):
    if not image_base64_data and not file_path:
        raise ValueError("Either image_base64_data or file_path must be provided.")

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    if not imgs:
        return [{"error": "No face with two eyes detected in the image."}]
    
    result = []
    for img in imgs:
        try:
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
            len_image_array = 32 * 32 * 3 + 32 * 32
            final = combined_img.reshape(1, len_image_array).astype(float)
            result.append({
                'class': class_number_to_name(__model.predict(final)[0]),
                'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
                'class_dictionary': __class_name_to_number
            })
        except Exception as e:
            result.append({"error": str(e)})
    return result


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/image_classification.pkl', 'rb') as f:
            __model = joblib.load(f)

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

# Streamlit App
st.title("Image Classification App")

# Load model artifacts
load_saved_artifacts()

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read and display the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # Perform classification
    try:
        st.text("Classifying image...")
        results = classify_image(img_base64)
        for result in results:
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"Class: {result['class']}")
                # st.write(f"Class Probability: {result['class_probability']}")
                # st.json(result['class_dictionary'])
                # Print probabilities with class names
                i = 0
                for key, values in result['class_dictionary'].items():
                    # st.write(f"{key}: {result['class_probability'][i]}")
                    result['class_dictionary'][key] = result['class_probability'][i]
                    i+=1
                st.json(result['class_dictionary'])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

