import streamlit as st
import cv2
import PIL
import numpy as np
import keras
import os
import random


IMG_SIZE = 224
emotion = { 0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprise" }

angry = ["Practice deep breathing, Practice meditation to calm the mind.", "Engage in physical exercise to release built-up tension.", "Take a break to cool off before addressing the source of anger.", "Focus on problem-solving rather than dwelling on the emotion."]
fear = ["Identify and challenge irrational thoughts through self-reflection.", "Seek support from a trusted friend or professional.", "Break down overwhelming tasks into smaller, manageable steps.", "Engage in relaxation techniques, such as deep breathing or meditation.", "Confront fears gradually to build confidence."]
happy = ["Spend time with loved ones.", "Play your favorite sport.", "Watch your favorite movie.", "Watch Freinds TV series.", "Watch Interstellar", "Listen to Taylor Swift.", "Listen to Justin Beiber."]
neutral = ["Explore a new hobby or activity.", "Take a walk or spend time in nature to clear the mind.", "Try a new restaurant to add variety.", "Reflect on personal goals and plan for the future.", "Connect with friends or family for a casual outing."]
sad = ["Reach out to friends or family for support.", "Engage in self-care activities, take a warm bath, practice mindfulness.", "Watch a comforting movie, read a favorite book.", "Take a gentle walk in nature."]
surprise = ["Embrace the moment and savor the experience.", "Share the surprising news or event with friends or family.", "Capture the moment with photos or a journal entry.", "Allow time for the surprise to settle before reacting."]

recommendation = { 0: angry, 1: happy, 2: neutral, 3: sad, 4: surprise }

model = keras.models.load_model("FER_Model.h5")

def prepare_image(image):
    image = keras.applications.resnet_v2.preprocess_input(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

def Predict():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # roi: region of interest
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+h]
        
        cropped_img = np.expand_dims(prepare_image(roi), axis=0)
        # predict the emotions
        emotion_prediction = model.predict(cropped_img, verbose=0)
        maxindex = np.argmax(emotion_prediction, axis=1)[0]
        cv2.putText(image, emotion[maxindex], (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        global rec
        rec = recommendation[maxindex][random.randint(0, len(recommendation[maxindex]) - 1)]




st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🟦",
)

st.title("Upload Image")

uploaded_file = st.file_uploader(label="Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = PIL.Image.open(uploaded_file)
    image = np.array(pil_image)
    Predict()

    st.image(image, channels="RGB", caption="Uploaded Image", use_column_width=True)
    st.markdown(f'<div style="border:1px solid black; padding:10px; font-weight: bold">{rec}</div>', unsafe_allow_html=True)

    filename = "processed_image.jpg"
    processed_image_path = "temp_files/" + filename
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(processed_image_path, image)
    
    processed_image = open(processed_image_path, "rb")
    download_btn = st.download_button(
            label="Download",
            data=processed_image,
            file_name=filename,
            key="download_button",
        )
    processed_image.close()
    os.remove(processed_image_path)



