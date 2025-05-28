import os
from utils.preprocessing import preprocess_lip_reading_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed, Conv2D, MaxPooling2D, GRU, Dense, Dropout



# Load Lip Reading Data
X, y, label_dict, label_list = preprocess_lip_reading_data(
    "./sign_language_ai_project/data/lip_reading/",
    sequence_length=5,
    img_size=20
)
# Build Correct Light LipNet Model
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(5, 50, 50, 1)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(GlobalAveragePooling2D()),  # 🔥 This replaces Flatten
    GRU(128, return_sequences=False),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

# Load Lip Reading Data
X, y, label_dict, label_list = preprocess_lip_reading_data(
    "./sign_language_ai_project/data/lip_reading/",
    sequence_length=5,
    img_size=20
)

# Build Light LipNet Model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# After loading data
print(f"Total training samples: {X.shape[0]}")

# Training
print("🚀 Starting LipNet model training...")
model.fit(X, y, epochs=15, batch_size=1, verbose=1)

# Save
model.save("./sign_language_ai_project/models/lipnet_model.h5")
print("✅ LipNet Model Trained and Saved Successfully!")
