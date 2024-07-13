import os
import numpy as np
import cv2
import tensorflow as tf

# Mengatur TensorFlow untuk menggunakan hanya CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Fungsi untuk memproses gambar
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image at path: {image_path}")
    img_resized = cv2.resize(img, (224, 224))  # Mengubah ukuran gambar sesuai dengan input yang diharapkan oleh ResNet
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # Membuat batch yang terdiri dari satu gambar
    img_preprocessed = tf.keras.applications.resnet.preprocess_input(img_batch)  # Preprocessing gambar sesuai kebutuhan ResNet
    return img, img_preprocessed

# Fungsi untuk memprediksi gambar dengan model
def predict_image(model_path, image_path):
    img, img_preprocessed = preprocess_image(image_path)
    model = tf.keras.models.load_model(model_path)  # Memuat model
    preds = model.predict(img_preprocessed)  # Memprediksi gambar
    return img, preds

# Fungsi untuk mendecode prediksi
def decode_predictions(preds, class_labels):
    class_id = np.argmax(preds)
    class_label = class_labels[class_id]
    class_probability = preds[0][class_id]
    return class_label, class_probability

def resize_image(image_path):
    # Membaca gambar dari file
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not open or find the image: " + image_path)

    # Mengubah ukuran gambar menjadi 224x224 piksel
    resized_img = cv2.resize(img, (640, 480))

    return resized_img

# Define the path to the model and the image
model_path = 'E:/IVAN M.T/COBA/coba/training2.h5'
image_path = "E:\\IVAN M.T\\COBA\\coba\\flower.png"  # Pastikan ini adalah file gambar yang benar
class_labels = ['flower', 'plastik', 'wood']  # Ganti dengan label kelas aktual Anda
resized_img = resize_image(image_path)
# Prediksi gambar
img, preds = predict_image(model_path, image_path)
label, probability = decode_predictions(preds, class_labels)

# Menambahkan teks prediksi ke gambar
font = cv2.FONT_HERSHEY_SIMPLEX
text = f"{label}: {probability * 100:.2f}%"
cv2.putText(resized_img, text, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Menampilkan gambar
cv2.imshow('Prediction', resized_img)
#cv2.imshow('Prediction', img_preprocessed)
cv2.waitKey(0)  # Menunggu tombol apapun ditekan
cv2.destroyAllWindows()  # Tutup jendela gambar
