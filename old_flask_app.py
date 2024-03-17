from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import glob
import pytz
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    encodeed_list = []
    images = glob.glob(path + '/*.jpg')

    for img in images:
        img = cv2.imread(f'{img}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeed_list.append(encoded_face)

    return encodeed_list

def newmarkAttendence(name, date_today):
    csv_file = f'Attendance_{date_today}.csv'

    # Set the timezone to Asia/Dhaka (BDT)
    timezone_bdt = pytz.timezone('Asia/Dhaka')

    with open(csv_file, 'a+') as f:
        # Check if the file is empty (new file)
        if os.path.getsize(csv_file) == 0:
            f.write('Name,Date,Time\n')

        # Read the existing data and update attendance
        f.seek(0)
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]

        #if name.lower() in nameList:
            #return f"{name} has already been marked present today ({date_today})."
        #else:
        current_time = datetime.now(timezone_bdt).strftime('%d-%B-%Y %I:%M:%S:%p')
        new_row = f'{name},{current_time}\n'
        f.writelines(new_row)
        show_row = f'{name},{current_time}'
        return show_row


def new_call_attendance(img_path, encoded_face_train, class_names):
    imgS = cv2.imread(f'{img_path}')
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Rest of your face recognition code...

    encoded_face = face_recognition.face_encodings(imgS)
    
    if not encoded_face:
        return "No face detected in the image."

    encoded_face = encoded_face[0]

    matches = face_recognition.compare_faces(encoded_face_train, encoded_face)

    if any(matches):
        faceDist = face_recognition.face_distance(encoded_face_train, encoded_face)
        matchIndex = np.argmin(faceDist)
        name = class_names[matchIndex].upper().lower()
        date_today = datetime.now().strftime('%d-%B-%Y')
        return newmarkAttendence(name, date_today)
    else:
        return "Person is not recognized."

@app.route('/', methods=['GET','POST'])
def attendance_api():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Create the 'uploads' directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # Save the uploaded file
        file_path = os.path.join(uploads_dir, file.filename)
        print("File Path:", file_path)
        file.save(file_path)

        # Set up your training data
        train_img_path = '/home/ptom/mysite/images'
        encoded_face_train = find_embeddings(train_img_path)
        class_names_list = class_names(train_img_path)

        # Provide the attendance response
        result = new_call_attendance(file_path, encoded_face_train, class_names_list)

        return jsonify({'result': result})

    # Handling GET requests
    return "Hello, this is the attendance API. Please use a POST request with a file."

if __name__ == "__main__":
    app.run(debug=True)
