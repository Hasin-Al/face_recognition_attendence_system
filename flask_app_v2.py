from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import glob
import pytz
from flask_cors import CORS
import csv
import uuid  # For generating API key tokens

app = Flask(__name__, template_folder='/home/hasinmanjare/mysite/')

CORS(app)

# Function to log API requests
def log_request(api_key, endpoint, company_name):
    # Define CSV file path
    csv_file = 'api_usage.csv'

    # Define field names
    fieldnames = ['DateTime', 'APIUser', 'CompanyName', 'Endpoint']

    # Check if file exists, if not, create it and write header
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()

        # Write data to CSV file
        writer.writerow({'DateTime': datetime.now(), 'APIUser': api_key, 'CompanyName': company_name, 'Endpoint': endpoint})

def generate_api_key():
    return str(uuid.uuid4())[:8]  # Generate a random UUID and truncate to 8 characters

def save_registration_info(company_name, email, api_key):
    # Define CSV file path
    csv_file = 'customer.csv'

    # Define field names
    fieldnames = ['CompanyName', 'Email', 'APIKey']

    # Write registration info to CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'CompanyName': company_name, 'Email': email, 'APIKey': api_key})

@app.route('/')
def home():
    return render_template('registration_form.html')

@app.route('/register', methods=['POST'])
def register():
    company_name = request.form['company_name']
    email = request.form['email']

    # Generate API key token
    api_key = generate_api_key()

    # Save registration info
    save_registration_info(company_name, email, api_key)

    # Create directories for company
    company_dir = os.path.join('mysite', company_name)
    train_img_path = os.path.join(company_dir, 'train_images')
    weight_files_path = os.path.join(company_dir, 'weight_files')

    os.makedirs(company_dir, exist_ok=True)
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(weight_files_path, exist_ok=True)

    return jsonify({'message': 'Registration successful!', 'APIKey': api_key})
    

def load_valid_api_keys():
    api_keys = {}
    with open('customer.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            api_keys[row['Email']] = row['APIKey']
    return api_keys

# Initialize valid API keys
valid_api_keys = load_valid_api_keys()

# Remaining routes and functions remain unchanged...
@app.before_request
def before_request():
    if request.endpoint != 'static':
        # Extract API key from the request headers
        api_key = request.headers.get('X-API-Key')
        if api_key:
            # Check if the API key is valid
            if api_key not in valid_api_keys.values():
                abort(401)  # Unauthorized error if API key is not valid

            # Extract company name from the request form data
            company_name = request.form.get('company_name') if request.method == 'POST' else None
            # Log the request
            log_request(api_key, request.endpoint, company_name)
        else:
            abort(401)  # Unauthorized error if no API key provided


# Dummy API keys (Replace with your actual API keys)






def extract_name(file_path):
    return file_path.split('/')[-1].split('.')[0]

def class_names(path):
    classes = []
    images = glob.glob(path + '/*.jpg')
    for image in images:
        name = extract_name(image)
        classes.append(name)
    return classes

def find_embeddings(path):
    encoded_list = []
    images = glob.glob(path + '/*.jpg')

    for img in images:
        img = cv2.imread(f'{img}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encoded_list.append(encoded_face)

    return encoded_list

def mark_attendance(name, date_today, company_name):
    csv_file = f'/home/hasinmanjare/mysite/{company_name}/Attendance_{date_today}.csv'

    timezone = pytz.timezone('Asia/Dhaka')

    with open(csv_file, 'a+') as f:
        if os.path.getsize(csv_file) == 0:
            f.write('Name,Date,Time\n')

        current_time = datetime.now(timezone).strftime('%d-%B-%Y %I:%M:%S:%p')
        new_row = f'{name},{current_time}\n'
        f.writelines(new_row)
        show_row = f'{name},{current_time}'
        return show_row

def authenticate_request():
    try:
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in valid_api_keys.values():
            abort(401)
    except Exception as e:
        print(f"Error during authentication: {str(e)}")
        abort(500)






@app.route('/upload_image', methods=['GET','POST'])
def upload_image():
    authenticate_request()
    if 'company_name' not in request.form:
        return jsonify({'error': 'Company name not provided'})

    company_name = request.form['company_name']
    train_img_path = f'/home/hasinmanjare/mysite/{company_name}/train_images'

    try:
        if not os.path.exists(train_img_path):
            os.makedirs(train_img_path)

        #uploaded_files = request.files.getlist('images')
        uploaded_files = request.files.values()
        for uploaded_file in uploaded_files:
            filename = uploaded_file.filename
            file_path = os.path.join(train_img_path, filename)
            uploaded_file.save(file_path)

        return jsonify({'message': 'Images uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/train', methods=['GET','POST'])
def train_model():
    authenticate_request()
    if 'company_name' not in request.form:
        return jsonify({'error': 'Company name not provided'})

    company_name = request.form['company_name']
    train_img_path = f'/home/hasinmanjare/mysite/{company_name}/train_images'
    weight_files_path = f'/home/hasinmanjare/mysite/{company_name}/weight_files'

    try:
        # Check if company directory exists
        if not os.path.exists(train_img_path):
            return jsonify({'error': 'Company name is not valid'})

        if not os.path.exists(weight_files_path):
            os.makedirs(weight_files_path)

        # Find embeddings and class names
        encoded_face_train = find_embeddings(train_img_path)
        class_names_list = class_names(train_img_path)

        # Save embeddings and class names
        np.save(f'{weight_files_path}/embeddings.npy', encoded_face_train)
        np.save(f'{weight_files_path}/classes.npy', class_names_list)

        return jsonify({'message': 'Training completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['GET','POST'])
def predict():
    authenticate_request()
    if 'file' not in request.files or 'company_name' not in request.form:
        return jsonify({'error': 'No file or company name provided'})

    file = request.files['file']
    company_name = request.form['company_name']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    weight_files_path = f'/home/hasinmanjare/mysite/{company_name}/weight_files'

    if not os.path.exists(weight_files_path):
        return jsonify({'error': 'Weight files not found for the company'})

    # Load embeddings and class names
    encoded_face_train = np.load(f'{weight_files_path}/embeddings.npy')
    class_names_list = np.load(f'{weight_files_path}/classes.npy')

    # Perform prediction
    result = predict_attendance(file, encoded_face_train, class_names_list, company_name)

    return jsonify({'result': result})

def predict_attendance(uploaded_file, encoded_face_train, class_names_list, company_name):
    imgS = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    encoded_face = face_recognition.face_encodings(imgS)

    if not encoded_face:
        return "No face detected in the image."

    encoded_face = encoded_face[0]

    matches = face_recognition.compare_faces(encoded_face_train, encoded_face)

    if any(matches):
        faceDist = face_recognition.face_distance(encoded_face_train, encoded_face)
        matchIndex = np.argmin(faceDist)
        name = class_names_list[matchIndex].upper().lower()
        date_today = datetime.now().strftime('%d-%B-%Y')
        return mark_attendance(name, date_today, company_name)
    else:
        return "Person is not recognized."

if __name__ == "__main__":
    app.run(debug=True)