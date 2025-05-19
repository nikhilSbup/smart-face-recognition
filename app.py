import cv2
import os
from flask import Flask, request, render_template,jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask import render_template, send_file

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Class,Time')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return face_points
    except Exception as e:
        print(f"Error in extract_faces: {e}")
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function that trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in the attendance folder
def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        names = df['Name']
        rolls = df['Roll']
        classes = df['Class']
        times = df['Time']
        l = len(df)
        return names, rolls, classes, times, l
    except KeyError:
        # Handle the case where 'Class' column is not present in the CSV file
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        names = df['Name']
        rolls = df['Roll']
        # Add a default value for 'Class' if not present
        classes = ['N/A'] * len(df)
        times = df['Time']
        l = len(df)
        return names, rolls, classes, times, l

#### Add Attendance of a specific user with class information
def add_attendance(name, user_class):
    if name:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")

        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if int(userid) not in list(df['Roll']):
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{user_class},{current_time}')

#### Get all users and their details
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    classes = []
    l = len(userlist)

    for i in userlist:
        name, roll, user_class = i.split('_')
        names.append(name)
        rolls.append(roll)
        classes.append(user_class)

    return userlist, names, rolls, classes, l

#### Delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)

    for i in pics:
        os.remove(duser + '/' + i)

    os.rmdir(duser)

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, classes, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, classes=classes, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)                    #0 for default webcam
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person:
                identified_person = identified_person[0]
                if identified_person:
                    # Retrieve the user class from the existing userlist
                    userlist, names, rolls, classes, _ = getallusers()
                    user_index = userlist.index(identified_person)
                    user_class = classes[user_index]
                    user_name = names[user_index]
                    user_roll = rolls[user_index]

                    add_attendance(identified_person, user_class)  # Pass the correct user_class
                    text_to_display = f'{user_name} - {user_roll} - {user_class}'
                    cv2.putText(frame, text_to_display, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)  # Draw a rectangle around the face
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, classes, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, classes=classes, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

# ...

#### This function will run when we click on Add a new user button
@app.route('/add', methods=['POST'])
def add():
    try:
        apass = request.form['admin_password']
        if apass == 'admin':
            newuser = request.form['newusername']
            newroll = request.form['newuserid']
            newclass = request.form['class_select']  # Retrieve user class from the form
            if newroll.isdigit():
                os.makedirs(f'static/faces/{newuser}_{newroll}_{newclass}')
                ret = True
                # Try with CAP_DSHOW
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                # Or try with CAP_MSMF
                # cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
                # Set properties (adjust values accordingly)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                count = 0
                while ret and count < 50:  # Change the condition to capture 50 images
                    ret, frame = cap.read()
                    faces = extract_faces(frame)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                        face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))

                        # Display additional information (name, class, etc.) on the frame
                        cv2.putText(frame, f'Capturing: {count}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                        cv2.putText(frame, f'Name: {newuser} | Roll: {newroll} | Class: {newclass}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

                        cv2.imwrite(f'static/faces/{newuser}_{newroll}_{newclass}/face_{count}.jpg', face)
                        count += 1
                    cv2.imshow('Get Images for Model Training', frame)
                    if cv2.waitKey(1) == 27:
                        break

                cap.release()
                cv2.destroyAllWindows()
                train_model()
                add_attendance(f'{newuser}_{newroll}_{newclass}', newclass)  # Pass the user class to add_attendance
                names, rolls, classes, times, l = extract_attendance()
                return render_template('home.html', names=names, rolls=rolls, classes=classes, times=times, l=l,
                                       totalreg=totalreg(), datetoday2=datetoday2, mess=f'User Added Successfully!')
            else:
                names, rolls, classes, times, l = extract_attendance()
                return render_template('home.html', names=names, rolls=rolls, classes=classes, times=times, l=l,
                                       totalreg=totalreg(), datetoday2=datetoday2, mess=f'Invalid Roll Number!')
        else:
            names, rolls, classes, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, classes=classes, times=times, l=l,
                                   totalreg=totalreg(), datetoday2=datetoday2,
                                   mess='Incorrect Admin Password! Please try again.')
    except Exception as e:
        names, rolls, classes, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, classes=classes, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='An error occurred. Please try again.')

# ...
###Redirecting to attendance dashboard

# Update the route to render the new HTML page
@app.route('/attendance_dashboard')
def attendance_dashboard():
    # Get all files in the Attendance directory
    attendance_files = os.listdir('Attendance')
    
    # Separate files into two groups: master attendance and regular attendance
    master_attendance_files = [file for file in attendance_files if 'Master' in file]
    regular_attendance_files = [file for file in attendance_files if 'Master' not in file]

    return render_template('attendance_dashboard.html', 
                           master_attendance_files=master_attendance_files, 
                           regular_attendance_files=regular_attendance_files)

# Route to serve CSV files
@app.route('/attendance_file/<filename>')
def get_attendance_file(filename):
    # Serve the requested CSV file
    return send_file(f'Attendance/{filename}', as_attachment=True)

##### creating master data

@app.route('/create_master_attendance', methods=['GET'])
def create_master_attendance():
    # Get the current month and year
    today = datetime.today()
    current_month = today.strftime("%m")  # Use two digits for month
    current_year = today.strftime("%y")   # Use last two digits for year

    # Assuming the 'Attendance' folder is in the same directory as your app.py
    attendance_folder = os.path.join(os.path.dirname(__file__), 'Attendance')

    # Get a list of all files in the 'Attendance' folder
    all_files = os.listdir(attendance_folder)

    # Filter files for the current month and year
    current_month_files = [f for f in all_files if f"Attendance-{current_month}" in f and current_year in f]

    # Combine data from all CSV files of the current month
    combined_data = []
    for current_month_file in current_month_files:
        file_path = os.path.join(attendance_folder, current_month_file)
        print(f"Reading File: {file_path}")
        daily_data = pd.read_csv(file_path)
        
        # Extract the date from the filename and remove the file extension
        date_str = current_month_file.split('-')[1].split('.')[0]

        # Check if the date string is not empty before attempting to convert
        if date_str:
            attendance_date = datetime.strptime(date_str, "%m_%d_%y").strftime("%Y-%m-%d")
    
            # Add a 'Date' column with the extracted date
            daily_data['Date'] = attendance_date
            combined_data.append(daily_data)
        else:
            print(f"Error: Unable to extract date from filename: {current_month_file}")


    # Concatenate all data into a single DataFrame
    if combined_data:
        master_data = pd.concat(combined_data, ignore_index=True)
        # Save the master data to a CSV file for the current month
        master_file_path = f'Attendance/Master_Attendance_{current_month}_{current_year}.csv'
        master_data.to_csv(master_file_path, index=False)
        return jsonify({'message': f'Master monthly attendance for {today.strftime("%B %Y")} created successfully!', 'master_file_path': master_file_path})
    else:
        return jsonify({'error': 'No valid CSV files found for the current month and year.'})

if __name__ == '__main__':
    app.run(debug=True)
