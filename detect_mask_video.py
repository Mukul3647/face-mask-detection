from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle

model = load_model("mask_detector.model")
lb = pickle.loads(open("label_encoder.pickle", "rb").read())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    face = cv2.resize(frame, (224, 224))
    face = face.astype("float32")
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    (mask, withoutMask) = model.predict(face)[0]
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (10, 10), (200, 60), color, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
