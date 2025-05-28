import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Preprocess Static Gesture Dataset
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
                    img = img / 255.0  # Normalize pixel values
                    images.append(img)
                    labels.append(label_dict[class_name])

    X = np.array(images)
    y = np.array(labels)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_val, y_train, y_val, label_dict
