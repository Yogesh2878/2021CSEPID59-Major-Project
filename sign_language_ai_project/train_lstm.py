# train_lstm.py
import os
from utils.preprocessing import preprocess_dynamic_gesture_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, Dropout

# Load Data
X, y, label_dict = preprocess_dynamic_gesture_data(
    "./sign_language_ai_project/data/raw/dynamic_gestures/",
    sequence_length=5,
    img_size=64
)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Build LSTM Model
model = Sequential([
    TimeDistributed(Flatten(), input_shape=(20, 64, 64, 3)),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
print("🚀 Starting LSTM model training...")
model.fit(X, y, epochs=10, batch_size=8, verbose=1)

# Save Model
model.save("./sign_language_ai_project/models/lstm_dynamic_model.h5")
print("✅ LSTM Model Trained and Saved Successfully!")
