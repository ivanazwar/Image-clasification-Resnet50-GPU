import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Specify the path to your locally saved model
model_path = 'path/to/your/model/trainingFine.h5'  # Adjust to where you've saved your model

# Load the model
model = tf.keras.models.load_model(model_path)
class_names = ['flower', 'plastik', 'wood']

def process_image(image_path):
    # Load the image from the local filesystem
    img = Image.open(image_path)
    original_img = img.copy()  # Save the original image for display
    img = img.resize((224, 224))  # Resize image to fit model expected input
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image as per model's requirement
    return img_array, original_img  # Return both processed and original images

def predict(image_tensor):
    # Predict with the model
    preds = model.predict(image_tensor)
    return preds

# Path to the local image file
image_path = 'path/to/your/image.jpg'  # Update this to the path of your local image
processed_image, original_img = process_image(image_path)  # Make sure to capture both returned values
prediction = predict(processed_image)

# Get the class with the highest probability
predicted_class_index = np.argmax(prediction, axis=-1)
predicted_class_name = class_names[predicted_class_index[0]]  # Map the index to the class name

# Output the prediction
print("Predicted probabilities:", prediction)
print("Predicted class name:", predicted_class_name)

# Display the image with prediction label
plt.imshow(original_img)
plt.title(f"Prediction: {predicted_class_name}")
plt.axis('off')  # Turn off axis
plt.show()