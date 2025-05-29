import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
base_dir = r'C:\Users\DAMINI\Desktop\CG PROJECT\fruit-disease-detection\apple_disease_classification'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validate')

# Print paths to verify
print("Training directory:", train_dir)
print("Validation directory:", validation_dir)

# Check if directories exist
if not os.path.isdir(train_dir):


    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.isdir(validation_dir):
    raise FileNotFoundError(f"Validation directory not found: {validation_dir}")

# Define Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices to verify the number of classes
print("Training classes:", train_generator.class_indices)
print("Validation classes:", validation_generator.class_indices)

# Adjust the number of classes based on the data generators
num_classes = len(train_generator.class_indices)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')  # Adjust output units based on your number of classes
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model in the native Keras format
model.save(r'fruit-disease-detection/apple_disease_detection_model.keras')
