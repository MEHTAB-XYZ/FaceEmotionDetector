import cv2
import numpy as np
from keras.models import model_from_json

# Load model architecture and weights
with open("my_model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("my_model.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(48, 48, 1)  # Ensure shape is (48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (p, q, r, s) in faces:
        face = gray[q:q+s, p:p+r]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        pred = model.predict(img)
        prediction_label = labels[np.argmax(pred)]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
        cv2.putText(im, prediction_label, (p, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Output", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
