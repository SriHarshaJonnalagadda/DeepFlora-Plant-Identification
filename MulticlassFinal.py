
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from flask import Flask, request, render_template, jsonify,redirect, url_for
#from flask_login import LoginManager, login_user, UserMixin, login_required
from werkzeug.utils import secure_filename
import os
import time
import mysql.connector


conn = mysql.connector.connect(
   host='localhost',
   user='root',
   password='Harsha@2801',
   database='medicinal_leaf')
cursor = conn.cursor()


def get_description(plant_name: str):
   command = f"SELECT Scientific_Name, General_Description, Benefits FROM medicinal_plants WHERE Common_Name = '{plant_name}'"
   cursor.execute(command)
   result = cursor.fetchone()
   return result


app = Flask(__name__,  static_folder='static')
#login_manager = LoginManager()
#login_manager.init_app(app)

# Load the saved multi-class CNN model
model = tf.keras.models.load_model('Multi_class_cnn.keras')
 # Define the path to the image you want to predict
def predict_image(image_path, model):
 # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image

 # Make a prediction
    predictions = model.predict(img)

 # Get the predicted class index (index with highest probability)
    predicted_class_index = np.argmax(predictions)

 # Optionally, map the predicted class index to class names
    class_labels = ['Rasna',
    'Arive-Dantu',
    'Jackfruit',
    'Neem',
    'Basale',
    'Indian Mustard',
    'Karanda',
    'Lemon',
    'Roxburgh fig',
    'Peepal Tree',
    'Hibiscus Rosa-sinensis',
    'Jasmine',
    'Mango',
    'Mint',
    'Drumstick',
    'Jamaica Cherry-Gasagase',
    'Curry',
    'Oleander',
    'Parijata',
    'Tulsi',
    'Betel',
    'Mexican Mint',
    'Indian Beech',
    'Guava',
    'Pomegranate',
    'Sandalwood',
    'Jamun',
    'Rose Apple',
    'Crape Jasmine',
    'Fenugreek'] # Replace with your class names
    predicted_class_name = class_labels[predicted_class_index]
    return predicted_class_name

@app.route('/')
def index():
    return render_template('introduction.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('new_hackathon1.html')

@app.route('/upload', methods=['POST'])
#@login_required
def upload():
    if request.method == 'POST':
        file = request.files['imagefile']

        # Define the directory where uploaded images will be stored
        upload_dir = r'C:\Users\harsh\OneDrive\Documents\PythonLearn\App\static\uploaded_images'  # Use backslashes and ensure folder names are consistent

        # Create a unique filename based on the current timestamp
        timestamp = str(int(time.time()))
        filename = secure_filename(timestamp + '_' + file.filename)
        file_path = os.path.join(upload_dir, filename)

        # Save the uploaded file with the unique filename
        file.save(file_path)

        # Predict the image and get the result
        predicted_class_name = predict_image(file_path, model)
        print("Predicted Class Name:", predicted_class_name)  # Add this line for debugging
        
        scientific_name, description, benifits = get_description(predicted_class_name)

        return render_template('prediction.html', predicted_class_name=predicted_class_name,scientific_name=scientific_name, description=description, benifits=benifits, filename=filename)

    # In case of a GET request or if the above conditions are not met, return a default response
    return "Invalid request."




'''return jsonify({'predicted_class': predicted_class_name})'''

if __name__ == '__main__':
    app.run(debug=True)
    cursor.close()



