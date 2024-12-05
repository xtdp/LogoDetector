# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras import Input

# Data_train_path = 'dataset\\train'
# Data_test_path = 'dataset\\test'
# Data_val_path = 'dataset\\validation'

# img_width = 250
# img_height = 250

# data_train = tf.keras.utils.image_dataset_from_directory(
#     Data_train_path,
#     shuffle=True,
#     image_size=(img_width, img_height),
#     batch_size=32,
#     validation_split=False
# )

# data_val = tf.keras.utils.image_dataset_from_directory(
#     Data_val_path,
#     shuffle=False,
#     image_size=(img_width, img_height),
#     batch_size=32,
#     validation_split=False
# )

# data_test = tf.keras.utils.image_dataset_from_directory(
#     Data_test_path,
#     shuffle=False,
#     image_size=(img_width, img_height),
#     batch_size=32,
#     validation_split=False
# )
# data_cat = data_train.class_names
# print(data_cat)

# data_augmentation = Sequential([
#     layers.RandomFlip("horizontal_and_vertical"),
#     layers.RandomRotation(0.2),
#     layers.RandomZoom(0.2),
# ])
# model = Sequential(
#     [
#     Input(shape=(img_width, img_height, 3)),
#     data_augmentation,
#     layers.Rescaling(1./255),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(128,3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(256,3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dropout(0.2),
#     layers.Dense(512, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(len(data_cat))
# ]
# )
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=10000,
#     decay_rate=0.9)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# epochs_size = 100
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(data_train.classes),
#     y=data_train.classes)
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# history = model.fit(data_train, validation_data=data_val, epochs=epochs_size, callbacks=[early_stopping], class_weight=class_weights)

# image = '26b7a6cd98fdaf281b6829dda3c83b96.jpg'
# image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
# img_arr = tf.keras.utils.array_to_img(image)
# img_bat = tf.expand_dims(img_arr,0)

# predict = model.predict(img_bat)
# score = tf.nn.softmax(predict)
# print(data_cat[np.argmax(score)], np.max(score)*100)
# model.save('predict.h5')










import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight

Data_train_path = 'dataset\\DataSet\\train'
Data_test_path = 'dataset\\DataSet\\test'
Data_val_path = 'dataset\\DataSet\\validation'

img_width = 250
img_height = 250

# Load the datasets
data_train = tf.keras.utils.image_dataset_from_directory(
    Data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)

data_val = tf.keras.utils.image_dataset_from_directory(
    Data_val_path,
    shuffle=False,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)

data_test = tf.keras.utils.image_dataset_from_directory(
    Data_test_path,
    shuffle=False,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)

# Class names (categories)
data_cat = data_train.class_names
print(data_cat)

# Data augmentation setup
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# Model architecture
model = Sequential(
    [
        Input(shape=(img_width, img_height, 3)),
        layers.Rescaling(1./255),
        data_augmentation,  # Data augmentation as a layer
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),  # Add batch normalization
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dropout(0.4),  # Increase dropout for regularization
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(len(data_cat), activation='softmax')  # Use softmax activation for classification
    ]
)


# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.2
)

# Compile the model with the learning rate schedule
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# Define callbacks (without ReduceLROnPlateau)

# Train the model (everything else remains the same)
history = model.fit(
    data_train.map(lambda x, y: (data_augmentation(x, training=True), y)),
    validation_data=data_val,
    epochs=50,
    
    
)

# Save the model in .h5 format
try:
    model.save('predict_model.keras')
    print("saved")  # Save as HDF5
except Exception as e:
    print("Error saving model in .h5 format:", e)

# # Predicting on a new image
image = '26b7a6cd98fdaf281b6829dda3c83b96.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image)  # Convert image to array
img_bat = tf.expand_dims(img_arr, 0)  # Expand dimensions to make it batch size 1

# # Make predictions
predict = model.predict(img_bat)
score = tf.nn.softmax(predict)
print(data_cat[np.argmax(score)], np.max(score) * 100)

