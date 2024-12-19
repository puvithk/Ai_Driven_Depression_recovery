import os.path
import pickle
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
import keras.src.layers
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model
from scipy.sparse import hstack, csr_matrix

class Sentimental:
    def __init__(self):
        pwd = os.path.dirname(__file__)
        with open(os.path.join(pwd, 'naive_bayes_model.pkl'), 'rb') as model_file:
            self.loaded_model = pickle.load(model_file)

        with open(os.path.join(pwd, 'vectorizer_cleaned.pkl'), 'rb') as vec_file_cleaned:
            self.loaded_vectorizer_cleaned = pickle.load(vec_file_cleaned)

        with open(os.path.join(pwd, 'vectorizer_selected.pkl'), 'rb') as vec_file_selected:
            self.loaded_vectorizer_selected = pickle.load(vec_file_selected)

    def find_time(self):
        text_time = ["morning", "noon", "night"]
        current_time = datetime.datetime.now()
        if current_time.hour < 12:
            return text_time[0]
        elif current_time.hour < 18:
            return text_time[1]
        else:
            return text_time[2]

    def predict_sentiment(self, text):
        cleaned_text_vector = self.loaded_vectorizer_cleaned.transform([text])
        time_vector = self.loaded_vectorizer_selected.transform([self.find_time()])
        combined_features = hstack([cleaned_text_vector, time_vector])
        print(f"Combined features shape: {combined_features.shape}")

        expected_features = 10001
        current_features = combined_features.shape[1]
        if current_features < expected_features:
            padding = csr_matrix((combined_features.shape[0], expected_features - current_features))
            combined_features = hstack([combined_features, padding])

        sentiment = self.loaded_model.predict(combined_features)[0]
        sentiment = {0: "negative", 1: "neutral", 2: "positive"}.get(sentiment, "unknown")
        return sentiment


class VideoModel:
    def __init__(self):
        pwd = os.path.dirname(__file__)
        self.model = model = tf.saved_model.load(f'{pwd}/model')
        self.face_cascade = cv2.CascadeClassifier(os.path.join(pwd, 'haarcascade_frontalface_default.xml'))
        self.emotion_labels = ["angry", "fear", "happy", "sad"]  # Adjust based on your model's output

    def predict_emotion(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=3)
            face_roi = face_roi / 255.0

            predictions = self.model.serve(face_roi)
            emotion_index = np.argmax(predictions)
            emotion = self.emotion_labels[emotion_index]
            return emotion


if __name__ == "__main__":
    sentimental = Sentimental()

    videoModel = VideoModel()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotion = videoModel.predict_emotion(frame)
        print(f"Emotion: {emotion}")
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(sentimental.predict_sentiment("I am happy"))
