from flask import Flask, render_template, request, redirect, url_for,session
import numpy as np
import cv2
from keras.models import load_model
from patchify import patchify, unpatchify
from metrics import IoU, IoU_coef, IoU_loss, dice_coef, dice_coef_loss, accuracy
import os
import uuid
from PIL import Image


app = Flask(__name__)
app.secret_key="shinakthhainaamiska"

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'


# Rest of your Flask app code...


model = None
patch_size = 512

# Load the model weights
def load_model_weights():
    global model
    model = load_model('./drs_unet.hdf5', custom_objects={'IoU_loss': IoU_loss, 'IoU_coef': IoU_coef, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss, 'accuracy': accuracy})
    model.summary()

load_model_weights()

# Define the CLAHE function
def clahe_equalized(imgs):    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

# Define the prediction function
def predict(image):
    global model, patch_size
    # Preprocess the image
    test = image[:, :, 1]  # Selecting green channel
    test = clahe_equalized(test)  # Applying CLAHE
    SIZE_X = (image.shape[1] // patch_size) * patch_size  # Getting size multiple of patch size
    SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))
    test = np.array(test)

    patches = patchify(test, (patch_size, patch_size), step=patch_size)  # Create patches(patch_sizexpatch_sizex1)

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype('float32')) / 255.
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)  # Predict on single patch
            predicted_patches.append(single_patch_prediction)
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size, patch_size))
    reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape)  # Join patches to form the whole image

     # Set the segmentation result to grey color
    reconstructed_image = np.where(reconstructed_image != 0, 255, reconstructed_image)

    return reconstructed_image

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

def generate_unique_filename(filename):
    # Generate a unique filename using UUID
    unique_filename = str(uuid.uuid4()) + '_' + filename
    return unique_filename

@app.route('/predict', methods=['POST'])
def upload():
    file = request.files['file']
    # Code for image preprocessing and model prediction

    # Generate unique filenames
    filename = generate_unique_filename(file.filename)
    predicted_filename = 'predicted_' + filename

    # Save the original and predicted images
    original_image_path = os.path.join('static/uploads', filename)
    file.save(original_image_path)

    predicted_image = predict(cv2.imread(original_image_path))

    predicted_image_path = os.path.join('static/uploads', predicted_filename)
    cv2.imwrite(predicted_image_path, predicted_image)

    # Convert original and predicted images to PNG
    pred_image_png = Image.open(predicted_image_path).convert("RGB")
    predicted_image_path_png = predicted_image_path.replace(".tif", ".png")
    pred_image_png.save(predicted_image_path_png, "PNG")

    orig_image_png = Image.open(original_image_path).convert("RGB")
    original_image_path_png = original_image_path.replace(".tif", ".png")
    orig_image_png.save(original_image_path_png, "PNG")

    # Store image paths in session variables
    session['image1_path'] = original_image_path_png
    session['image2_path'] = predicted_image_path_png
    return redirect(url_for('result'))


    # Redirect to the result route with the PNG image paths
    # return redirect(url_for('result', image1=original_image_path_png, image2=predicted_image_path_png))



@app.route('/result')
def result():
    # Retrieve the image paths from the query parameters
    # Retrieve image paths from session variables
    image1_path = session.get('image1_path')
    image2_path = session.get('image2_path')
    return render_template('result.html', image1_path=image1_path, image2_path=image2_path)


if __name__ == '__main__':
    app.run(debug=True)
