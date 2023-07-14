from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import skimage.io
from keras.models import load_model
from patchify import patchify, unpatchify
from metrics import IoU, IoU_coef, IoU_loss, dice_coef, dice_coef_loss, accuracy
from urllib.parse import urlencode
import os
import uuid
from PIL import Image


app = Flask(__name__, static_url_path='/static')
app.secret_key="shinakthhainaamiska"

# Set the upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
app.config['STATIC_FOLDER'] = os.path.join(app.root_path, 'static')

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

def convert_to_png(tiff_path, png_path):
    image = Image.open(tiff_path)
    image.save(png_path, 'PNG')
    

@app.route('/predict', methods=['POST'])
def upload():
    file = request.files['file']
    # Code for image preprocessing and model prediction

    # Generate unique filenames
    filename = generate_unique_filename(file.filename)
    predicted_filename = 'predicted_' + filename

    # Save the original and predicted images
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(original_image_path)

    predicted_image = predict(cv2.imread(original_image_path))

    predicted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_filename)
    cv2.imwrite(predicted_image_path, predicted_image)
    
    #converting paths and images to png
    orig=original_image_path+".png"
    pred=predicted_image_path+".png"

    convert_to_png(original_image_path, orig)
    convert_to_png(predicted_image_path, pred)

    # return redirect(url_for('show_result',image_paths=image_paths))
    image1 = "uploads/" + filename + ".png"
    image2 = "uploads/predicted_" + filename + ".png"

    # Redirect to the /show_result route with the image paths as URL query parameters
    return redirect('/show_result?image1=' + image1 + '&image2=' + image2)

@app.route('/show_result')
def show_result():
    # Retrieve the image paths from the URL parameters
    
    image1=request.args.get('image1')
    image2=request.args.get('image2')

    # Render the result.html template and pass the image paths
    return render_template('result.html', image1=image1, image2=image2)

    
@app.route('/result')
def result():
    image1 = request.args.get('image1')
    image2 = request.args.get('image2')
    return render_template('result.html', image1=image1, image2=image2)





if __name__ == '__main__':
    app.run(debug=True)
