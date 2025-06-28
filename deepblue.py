import os
import re
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Define dataset path
DATASET_PATH = r"C:\Users\msdak\Documents\Celeb-FBI Dataset"

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Regular expression to extract labels from filenames
filename_pattern = re.compile(r"(\d+)_(\d+\.\d+)h_(\d+)w_(male|female)_(\d+)a")

# Function to extract labels from filename
def extract_labels(filename):
    match = filename_pattern.match(filename)
    if match:
        _, height, weight, gender, age = match.groups()
        height = float(height)  
        weight = int(weight)  
        gender = 1 if gender == "male" else 0  
        age = int(age)  
        return height, weight, gender, age
    return None

# Load dataset
images, labels = [], []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        label = extract_labels(filename)
        if label:
            img_path = os.path.join(DATASET_PATH, filename)
            img = load_img(img_path, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0  # Normalize
            images.append(img)
            labels.append(label)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save extracted labels to a text file
with open("extracted_labels.txt", "w") as f:
    for lbl in labels:
        f.write(f"{lbl}\n")

# Splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Build Enhanced Model
def build_model():
    base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze base model initially

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation="linear")(x)  # Predict height, weight, gender, age

    model = Model(inputs, outputs)
    return model, base_model

# Custom Metrics
def gender_accuracy(y_true, y_pred):
    y_true_gender = tf.round(y_true[:, 2])
    y_pred_gender = tf.round(y_pred[:, 2])
    return tf.keras.metrics.binary_accuracy(y_true_gender, y_pred_gender)

def percentage_within_margin(y_true, y_pred, margin=0.1):
    height_acc = tf.reduce_mean(tf.cast(tf.abs(y_true[:, 0] - y_pred[:, 0]) <= margin * y_true[:, 0], tf.float32))
    weight_acc = tf.reduce_mean(tf.cast(tf.abs(y_true[:, 1] - y_pred[:, 1]) <= margin * y_true[:, 1], tf.float32))
    age_acc = tf.reduce_mean(tf.cast(tf.abs(y_true[:, 3] - y_pred[:, 3]) <= margin * y_true[:, 3], tf.float32))
    return (height_acc + weight_acc + age_acc) / 3.0

# Create model
model, base_model = build_model()

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mae",  # Mean Absolute Error
              metrics=[gender_accuracy, lambda y_true, y_pred: percentage_within_margin(y_true, y_pred, 0.1)])

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# Train Model (Phase 1: Train Head Layers Only)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])

# Unfreeze base model and fine-tune (Phase 2)
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="mae",
              metrics=[gender_accuracy, lambda y_true, y_pred: percentage_within_margin(y_true, y_pred, 0.1)])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])

# Save Model
model.save("enhanced_height_weight_gender_age_model.h5")

# Evaluate Model
loss, gender_acc, overall_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Gender Accuracy: {gender_acc * 100:.2f}%, Overall Accuracy (within 10% margin): {overall_acc * 100:.2f}%")