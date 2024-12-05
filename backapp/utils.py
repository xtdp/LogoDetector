import tensorflow as tf
import numpy as np
import cv2

def preprocess_image(image_path, model_path="C:\\Users\\parth\\OneDrive\\Desktop\\predict_model.keras"):
    model = tf.keras.models.load_model(model_path)
    
    data_cat = ["real", "fake"]
    img_width = 250
    img_height = 250
    
    image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)
    
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    
    return np.argmax(score) == 1, np.max(score) * 100 
