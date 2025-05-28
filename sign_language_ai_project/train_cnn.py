import os
import numpy as np
import cv2
from utils.preprocessing import preprocess_static_gesture_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Preprocess Static Gesture Data
def preprocess_static_gesture_data(data_dir, img_size=64):
    images = []
    labels = []

    classes = os.listdir(data_dir)
    classes.sort()
    label_dict = {class_name: idx for idx, class_name in enumerate(classes)}

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(label_dict[class_name])

    X = np.array(images)
    y = np.array(labels)

    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_val, y_train, y_val, label_dict

# 2. Set dataset path
dataset_path = "./sign_language_ai_project/data/raw/static_gestures/"

# 3. Preprocess
X_train, X_val, y_train, y_val, label_dict = preprocess_static_gesture_data(dataset_path, img_size=64)

print(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples.")

# 4. Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train
print("🚀 Starting CNN training...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 6. Save the model
save_path = "./sign_language_ai_project/models/cnn_hand_model.h5"
model.save(save_path)
print(f"✅ CNN model trained and saved at {save_path}")
