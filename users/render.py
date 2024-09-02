from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings
import os
import cv2
from keras.utils import normalize
import keras.backend as K
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)


def predictor(uploaded_file, select_mra_or_cta, select_label):
    fs = FileSystemStorage()
    image_path = os.path.join(fs.location, fs.save(uploaded_file.name, uploaded_file))

    SIZE_X = 128 
    SIZE_Y = 128

    train_images = []

    img = cv2.imread(image_path, 0)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_images.append(img)

    train_images = np.array(train_images)
    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)
    X_test = train_images

    if select_mra_or_cta == "MRA" and select_label == "multi":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'multi_mr_roi.h5')
    elif select_mra_or_cta == "MRA" and select_label == "single":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'single_mr_roi.h5')
    elif select_mra_or_cta == "CTA" and select_label == "multi":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'multi_ct_roi.h5')
    else:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'single_ct_roi.h5')

    custom_objects = {'dice_coef': dice_coef}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
        
    output_directory = os.path.join(settings.MEDIA_ROOT, "predicted_image")
    os.makedirs(output_directory, exist_ok=True) 
    
    test_img = X_test[0]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]
    output_path = os.path.join(output_directory, f"predicted_image.png")
    plt.axis('off')
    plt.imshow(predicted_img, cmap='gray')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()

    return image_path, output_path