import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('fruit-disease-detection/apple_disease_detection_model.keras')

# Define the test data directory
test_data_dir = r'C:\Users\DAMINI\Desktop\CG PROJECT\fruit-disease-detection\apple_disease_classification\test'


# Preprocess the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f'Test accuracy: {test_accuracy}')
print(f'Test loss: {test_loss}')
