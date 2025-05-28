import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained models
cnn_model = tf.keras.models.load_model('./sign_language_ai_project/models/cnn_hand_model.h5')
lipnet_model = tf.keras.models.load_model('./sign_language_ai_project/models/lipnet_model.h5')

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Label dictionaries (update if needed)
hand_labels = {0: 'A', 1: 'B', 2: 'C'}  # Example: map output indices to your trained letters
lip_labels = {0: 'HELLO', 1: 'THANKS', 2: 'YES', 3:'BYE', 4:'A',5:'you_34'}  # Example: map output indices to your lip classes

# Preprocessing Functions
def preprocess_hand_image(hand_img):
    hand_img = cv2.resize(hand_img, (64, 64))
    hand_img = hand_img / 255.0
    hand_img = np.expand_dims(hand_img, axis=0)
    return hand_img

def preprocess_lip_frames(frames, sequence_length=5, img_size=50):
    processed_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame / 255.0
        processed_frames.append(frame)
    processed_frames = np.array(processed_frames)
    processed_frames = processed_frames.reshape(1, sequence_length, img_size, img_size, 1)
    return processed_frames

# Initialize capture
cap = cv2.VideoCapture(0)

lip_frame_buffer = []  # To collect frames for lip reading
SEQUENCE_LENGTH = 5  # Collect 5 frames for lip prediction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    hand_predicted_label = ''
    lip_predicted_label = ''

    # Hand Detection and Prediction
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get bounding box
            img_h, img_w, _ = frame.shape
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * img_w
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * img_w
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * img_h
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * img_h

            x_min = int(max(0, x_min - 20))
            x_max = int(min(img_w, x_max + 20))
            y_min = int(max(0, y_min - 20))
            y_max = int(min(img_h, y_max + 20))

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                preprocessed_hand = preprocess_hand_image(hand_img)
                prediction = cnn_model.predict(preprocessed_hand)
                predicted_class = np.argmax(prediction)
                hand_predicted_label = hand_labels.get(predicted_class, '')

            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Face and Lip Detection
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Get mouth landmarks (example: use 13, 14, 78, 308 for mouth bounding box)
            img_h, img_w, _ = frame.shape
            mouth_points = [13, 14, 78, 308]
            x = [face_landmarks.landmark[i].x * img_w for i in mouth_points]
            y = [face_landmarks.landmark[i].y * img_h for i in mouth_points]

            x_min = int(max(0, min(x) - 20))
            x_max = int(min(img_w, max(x) + 20))
            y_min = int(max(0, min(y) - 20))
            y_max = int(min(img_h, max(y) + 20))

            mouth_img = frame[y_min:y_max, x_min:x_max]
            if mouth_img.size > 0:
                lip_frame_buffer.append(mouth_img)

                if len(lip_frame_buffer) == SEQUENCE_LENGTH:
                    preprocessed_lips = preprocess_lip_frames(lip_frame_buffer)
                    prediction = lipnet_model.predict(preprocessed_lips)
                    predicted_class = np.argmax(prediction)
                    lip_predicted_label = lip_labels.get(predicted_class, '')
                    lip_frame_buffer = []  # Reset buffer after prediction

            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Display Predictions
    text = f"Hand Sign: {hand_predicted_label} | Lip Reading: {lip_predicted_label}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show
    cv2.imshow('Real-Time Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
