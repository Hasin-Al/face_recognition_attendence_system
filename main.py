from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import glob
import pytz

app = Flask(__name__)

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

def mark_attendance(name, date_today):
    csv_file = f'Attendance_{date_today}.csv'

    # Create the CSV file if it doesn't exist
    with open(csv_file, 'a+') as f:
        # Check if the file is empty (new file)
        if os.path.getsize(csv_file) == 0:
            f.write('Name,Time')

        # Read the existing data and update attendance
        f.seek(0)
        my_data_list = f.readlines()
        name_list = [entry.split(',')[0] for entry in my_data_list]

    # Set the timezone to Asia/Dhaka (BDT)
    timezone_bdt = pytz.timezone('Asia/Dhaka')

    if name.lower() in name_list:
        return f"{name} has already been marked present today ({date_today})."
    else:
        with open(csv_file, 'a') as f:
            time_now = datetime.now(timezone_bdt).strftime('%I:%M:%S:%p')
            f.writelines(f'\n{name}, {time_now}')
            return f"Attendance marked for {name} on {date_today} at {time_now}."

def call_attendance(img_path):
    imgS = cv2.imread(f'{img_path}')
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Rest of your face recognition code...

    encoded_face = face_recognition.face_encodings(imgS)[0]

    matches = face_recognition.compare_faces([encoded_face_train], encoded_face)
    face_dist = face_recognition.face_distance(encoded_face_train, encoded_face)
    match_index = np.argmin(face_dist)

    if matches:
        name = class_names[match_index].upper().lower()
        date_today = datetime.now().strftime('%d-%B-%Y')
        return mark_attendance(name, date_today)

@app.route('/api/attendance', methods=['POST'])
def attendance_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_path = 'uploads/' + file.filename
    file.save(file_path)

    result = call_attendance(file_path)

    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
