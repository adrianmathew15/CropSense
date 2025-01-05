from flask import Flask, render_template, request, Markup, jsonify

import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.nutrient import nutrient_dic

from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import traceback

import subprocess
import os
from flask import redirect
import psutil

#LOADING THE TRAINED MODELS 
#Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Custom functions for calculations
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


#Loading nutrient model
try:
    model = load_model('utils/nutrient_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")

# Prediction function
def predict_nutrient(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    classes = ['Nitrogen', 'Phosphorus', 'Potassium']
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence


#FLASK APP 
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Set a secret key for session management

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

@app.before_first_request
def create_tables():
    db.create_all()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action_type = request.form.get('actionType')  # Get the hidden field value
        if action_type == 'signup':  # Sign up form submission
            try:
                name = request.form['name']
                email = request.form['email']
                password = request.form['password']

                if not name or not email or not password:
                    return render_template('signup.html', message="All fields are required.")

                # Check if the email already exists
                existing_user = User.query.filter_by(email=email).first()
                if existing_user:
                    return render_template('signup.html', message="Email already registered! Please log in.")

                # Add new user
                new_user = User(name=name, email=email, password=password)
                db.session.add(new_user)
                db.session.commit()
                return render_template('signup.html', message="Sign up successful! Please log in.")
            except Exception as e:
                db.session.rollback()  # Rollback transaction on error
                print(f"Error occurred: {e}")
                return render_template('signup.html', message=f"An error occurred: {str(e)}")
            finally:
                db.session.close()  # Ensure session is closed
        elif action_type == 'login':  # Login form submission
            email = request.form['email']
            password = request.form['password']

            user = User.query.filter_by(email=email).first()
            if user and user.password == password:
                session['user_id'] = user.id  # Store user ID in session
                session['user_name'] = user.name  # Store user name in session
                return redirect(url_for('home'))  # Redirect to home page
            else:
                return render_template('signup.html', message="Invalid credentials. Please try again.")
    return render_template('signup.html')





@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        return f"Hello, {session['user_name']}! Welcome to your dashboard."
    else:
        return redirect(url_for('index'))
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user_id if it exists
    session.pop('user_name', None)  # Remove user_name if it exists
    return redirect(url_for('index'))  # Redirect to the login page

# @app.route('/logout')
# def logout():
#     session.clear()  # Clear session data
#     return redirect(url_for('login'))  # Redirect to login page


# render home page
@ app.route('/home')
def home():
    title = 'Crop Sense - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop Sense - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page
@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Crop Sense - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# Route for nutrient deficiency detection
@app.route('/nutrient')
def nutrient_deficiency():
    title = 'Crop Sense - Nutrient Deficiency Detection'
    return render_template('nutrient.html', title=title)

def is_port_in_use(port):
    """Check if a given port is in use."""
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            return True
    return False

@app.route('/webapp')
def launch_webapp():
    try:
        # Ensure the correct path to the webapp directory
        webapp_path = os.path.abspath(os.path.join(os.getcwd(), 'webapp'))
        run_script = os.path.join(webapp_path, 'run.py')

        if not os.path.isfile(run_script):
            return f"Error: Script '{run_script}' does not exist."

        # Ensure port 8080 is not in use
        if not is_port_in_use(8080):
            subprocess.Popen(["python", run_script], cwd=webapp_path, shell=True)

        # Redirect to run.py's application
        return redirect("http://127.0.0.1:8080")
    except Exception as e:
        return f"Error launching WebApp: {e}"





# render disease prediction input page

# RENDER PREDICTION PAGES
# render crop recommendation result page
@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop Sense - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page
@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Crop Sense - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Crop Sense - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# # Dictionary with detailed nutrient information
# nutrient_info = {
#     'Nitrogen': {
#         'description': 'Nitrogen is essential for the growth of plants and plays a key role in photosynthesis.',
#         'recommendation': 'Increase nitrogen-rich fertilizers such as urea, ammonium nitrate, etc.'
#     },
#     'Phosphorus': {
#         'description': 'Phosphorus is crucial for energy transfer and photosynthesis in plants.',
#         'recommendation': 'Use phosphorus-rich fertilizers such as superphosphate or bone meal.'
#     },
#     'Potassium': {
#         'description': 'Potassium helps plants build resistance to diseases and aids in root development.',
#         'recommendation': 'Use potassium-rich fertilizers like potassium chloride or potassium sulfate.'
#     }
# }

@app.route('/nutrient', methods=['GET', 'POST'])
def nutrient_prediction():
    title = 'Crop Sense - Nutrient Deficiency Detection'

    if request.method == 'GET':
        return render_template('nutrient.html', title=title)

    if request.method == 'POST':
        try:
            # Collect the uploaded image for nutrient analysis
            if 'image' not in request.files:
                raise ValueError("No file part in the request")
            
            file = request.files['image']
            if file.filename == '':
                raise ValueError("No file selected for uploading")
            
            # Save the uploaded image temporarily
            image_path = 'temp_image.jpg'
            file.save(image_path)

            # Call the prediction function for nutrient deficiency
            predicted_class, confidence = predict_nutrient(image_path)

            # Get detailed information from nutrient_dic for the predicted nutrient
            nutrient_details = nutrient_dic.get(predicted_class, "No details available.")
            
            # Use Markup to format the result directly and pass it to the result page
            result = Markup(f"""
                <p><b>{predicted_class}</b></p>
                <p><i>{nutrient_details}</i></p>
                <p><b>Confidence:</b> {confidence:.2f}%</p>
            """)

            # Render the result template with formatted result
            return render_template(
                'nutrient-result.html',
                result=result,  # Pass the formatted result directly
                title=title
            )
        except Exception as e:
            print(f"Error: {e}")
            return render_template(
                'nutrient.html',
                title=title,
                error="An error occurred while processing the image. Please try again."
            )


if __name__ == '__main__':
    app.run(debug=True)
    


