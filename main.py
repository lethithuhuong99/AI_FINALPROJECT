import cv2
from flask import Flask, Response, render_template, request, session, redirect, url_for
import os
import datetime
from datetime import timedelta
import time
import pandas as pd
import warnings
import tkinter as tk
from screeninfo import get_monitors

warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import hashlib
from threading import Thread
from flask_login import login_user, logout_user, current_user, login_required

import pymongo
import csv
import sys

# Use Flask
app = Flask(__name__, template_folder='html')

# Use Database with MongoDb
myclient = pymongo.MongoClient(
    "mongodb+srv://admin:admin123@cluster0.9n8yb.mongodb.net/employees?retryWrites=true&w=majority")
mydb = myclient['employees']
mycol = mydb['attendancedbs']
# delete all attendance
# mycol.delete_many({})
listEmployeesCol = mydb['employeedbs']
countersCol = mydb['counters']
admin = mydb['admin']
positions = [
    'CEO',
    'Director',
    'Deputy',
    'Chief Executive Officer',
    'Chief Information Officer',
    'Head of department',
    'Deputy of department',
    'Secterary',
    'Employee',
    'Trainee']

if (countersCol.count() == 0):
    countersCol.insert_one(
        {
            '_id': "activities",
            'seq': 0
        }
    )


def getNextSequence(name):
    ret = countersCol.find_and_modify(
        query={'_id': name},
        update={'$inc': {'seq': 1}},
        new=True
    )
    return ret


@app.route('/')
def list_employees():
    if 'username' in session:
        resultPois = listEmployeesCol.find()
        listEmp = [];
        for x in resultPois:
            listEmp.append(x)
        return render_template("listEmployees.html", len=len(listEmp), listEmp=listEmp, session=session)
    return redirect('/login')


@app.route('/list-attendance')
def list_attendance():
    if 'username' in session:
        attendance = mycol.aggregate([
            {
                "$lookup":
                    {
                        "from": "employeedbs",
                        "localField": "userId",
                        "foreignField": "id",
                        "as": "empDetail"
                    }
            },
        ])
        listAttendance = [];
        for x in attendance:
            listAttendance.append(x)
        return render_template("listAttendance.html", len=len(listAttendance), listAttendance=listAttendance)
    return redirect('/login')


@app.route('/update-employee/<id>')
def update_employee(id):
    if 'username' in session:
        empUpId = {"id": id}
        empUp = listEmployeesCol.find(empUpId)
        for x in empUp:
            empUpInfor = x

        # you can use the the rowData from template
        return render_template('updateEmployee.html', empUpInfor=empUpInfor, positions=positions)
    return redirect('/login')


@app.route('/updated-employee/<id>', methods=['GET', 'POST'])
def updated_employee(id):
    if 'username' in session:
        if request.method == 'POST':
            employeeInfor = {"$set": {
                'name': request.form.get('name'),
                'email': request.form.get('email'),
                'gender': request.form.get('gender'),
                'phoneNumber': request.form.get('phoneNumber'),
                'address': request.form.get('address'),
                'dateOfBirth': request.form.get('dateOfBirth'),
                'position': request.form.get('position'),
                'id': id,
            }}

            empUpId = {"id": id}

            existing_phoneNumber = listEmployeesCol.find_one({'phoneNumber': request.form['phoneNumber']})
            existing_email = listEmployeesCol.find_one({'email': request.form['email']})

            isExist = False
            if (existing_phoneNumber):
                if (existing_phoneNumber['id'] != str(id)):
                    isExist = True
            if (existing_email):
                if (existing_email['id'] != str(id)):
                    isExist = True
            if (isExist == True):
                if 'username' in session:
                    empUpId = {"id": id}
                    empUp = listEmployeesCol.find(empUpId)
                    for x in empUp:
                        empUpInfor = x

                    return render_template('updateEmployee.html', existingEmp=True, positions=positions,
                                           empUpInfor=empUpInfor)
            else:
                listEmployeesCol.update_one(empUpId, employeeInfor)
        # empUp = listEmployeesCol.find(empUpId)
        # for x in empUp:
        #     empUpInfor = x

        # you can use the the rowData from template
        return redirect('/')
    return redirect('/login')


@app.route('/delete-employee/<id>')
def delete_employee(id):
    if 'username' in session:
        empDelId = {"id": id}
        listEmployeesCol.delete_one(empDelId)
        # you can use the the rowData from template
        return redirect('/')
    return redirect('/login')


# Load model Mask
model = load_model('MyTrainingModel.h5')
threshold = 0.90


def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def get_className(classNo):
    if classNo == 0:
        return "Mask"
    elif classNo == 1:
        return "No Mask"


def get_text_size(text, textfont):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.getTextSize(text, font, textfont, 2)[0]


# Attendance Function
def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel" + os.sep + "Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("EmployeeIds" + os.sep + "EmployeeIds.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Date', 'Id', 'Mask', 'Checkin', 'Checkout']
    attendance = pd.DataFrame(columns=col_names)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    empInformation = {'Id': '', 'name': '', 'gender': '', 'dateOfBirth': '', 'position': '', 'mask': ''}

    # screen_width = get_monitors()[0].width
    # screen_height = get_monitors()[0].height
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    boxWidth = round(screen_width / 6)
    boxHeight = round(screen_height / 5)
    fontSizeBig = round(boxHeight / (boxWidth*1.5), 2)
    fontSize = round(round(screen_height / 8) / round(screen_width / 5), 2)
    space = round(boxHeight / 10)
    spaceX = round(boxWidth / 10)
    # print("screen_width", screen_width, "screen_height", screen_height)
    while True:
        ret, im = cam.read()
        im = cv2.flip(im, 1)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        now = datetime.datetime.now()
        hourStartCheckin = int(sys.argv[1])
        minuteStartCheckin = int(sys.argv[2])
        hourStartCheckout = int(sys.argv[4])
        minuteStartCheckout = int(sys.argv[5])

        startCheckIn = now.replace(hour=hourStartCheckin, minute=minuteStartCheckin, second=0, microsecond=0)
        # endCheckIn = now.replace(hour=hour, minute=minute, second=20, microsecond=0)
        endCheckIn = (startCheckIn + timedelta(hours=0, minutes=int(sys.argv[3])))
        startCheckOut = now.replace(hour=hourStartCheckout, minute=minuteStartCheckout, second=0, microsecond=0)
        # endCheckOut = now.replace(hour=hour, minute=minute, second=50, microsecond=0)
        endCheckOut = (startCheckOut + timedelta(hours=0, minutes=int(sys.argv[6])))

        # print("startCheckIn: ", startCheckIn, "endCheckIn: ", endCheckIn, 'startCheckOut: ', startCheckOut, 'endCheckOut', endCheckOut)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        if (empInformation['Id'] != ''):
            cv2.rectangle(im, (5, 5), (boxWidth, boxHeight), (0, 0, 0), -1)
            cv2.putText(im, 'HAVE A NICE DAY!',
                        (5 + round((boxWidth - get_text_size('HAVE A NICE DAY!', fontSizeBig)[0]) / 2), 20 + space * 8),
                        font,
                        fontSizeBig,
                        (0, 255, 255), 2)

            if (now < endCheckIn and now > startCheckIn):
                cv2.putText(im, 'WELCOME!',
                            (round((boxWidth - get_text_size('WELCOME!', fontSizeBig)[0]) / 2), 5 + space), font,
                            fontSizeBig, (0, 255, 255), 2)
            else:
                if (now < startCheckOut):
                    empInformation = {'Id': '', 'name': '', 'gender': '', 'dateOfBirth': '', 'position': '', 'mask': ''}
            if (now > startCheckOut and now < endCheckOut):
                cv2.putText(im, 'GOOD BYE!',
                            (round((boxWidth - get_text_size('GOOD BYE!', fontSizeBig)[0]) / 2), 5 + space), font,
                            fontSizeBig, (0, 255, 255), 2)
            else:
                if (now >= endCheckOut):
                    empInformation = {'Id': '', 'name': '', 'gender': '', 'dateOfBirth': '', 'position': '', 'mask': ''}
        cv2.putText(im, empInformation['Id'], (spaceX, get_text_size(empInformation['Id'], fontSizeBig)[1] + space * 2),
                    font,
                    fontSize, (255, 255, 255), 1)
        cv2.putText(im, empInformation['name'],
                    (spaceX, get_text_size(empInformation['name'], fontSizeBig)[1] + space * 3), font,
                    fontSize, (255, 255, 255), 1)
        cv2.putText(im, empInformation['gender'],
                    (spaceX, get_text_size(empInformation['gender'], fontSizeBig)[1] + space * 4),
                    font, fontSize, (255, 255, 255), 1)
        cv2.putText(im, empInformation['dateOfBirth'],
                    (spaceX, get_text_size(empInformation['dateOfBirth'], fontSizeBig)[1] + space * 5), font, fontSize,
                    (255, 255, 255), 1)
        cv2.putText(im, empInformation['position'],
                    (spaceX, get_text_size(empInformation['position'], fontSizeBig)[1] + space * 6),
                    font, fontSize, (255, 255, 255), 1)
        cv2.putText(im, empInformation['mask'],
                    (spaceX, get_text_size(empInformation['mask'], fontSizeBig)[1] + space * 7), font,
                    fontSize, (255, 255, 255), 1)

        if ((now < endCheckIn and now > startCheckIn) or (now > startCheckOut and now < endCheckOut)):
            isExport = False
            isFace = False
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                isFace = True
                crop_img = im[y:y + h, x:x + h]
                img = cv2.resize(crop_img, (32, 32))
                img = preprocessing(img)
                img = img.reshape(1, 32, 32, 1)
                prediction = model.predict(img)
                classIndex = model.predict_classes(img)
                probabilityValue = np.amax(prediction)

                if probabilityValue > threshold:
                    if classIndex == 0:
                        # print("Mask")
                        empInformation['mask'] = 'Mask: Yes'
                    elif classIndex == 1:
                        # print("No Mask")
                        empInformation['mask'] = 'Mask: No'

                if (100 - conf) > 50:
                    # l???y t??n v?? id
                    confstr = "  {0}%".format(round(100 - conf))

                    empInfor = listEmployeesCol.find_one({'id': str(Id)})

                    if (empInfor != None):
                        empInformation['Id'] = 'Id: ' + str(Id)
                        empInformation['name'] = 'Name: ' + empInfor['name']
                        empInformation['gender'] = 'Gender: ' + empInfor['gender']
                        empInformation['dateOfBirth'] = 'DOB: ' + empInfor['dateOfBirth']
                        empInformation['position'] = 'Position: ' + empInfor['position']

                    # process attendance, save to file
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    # aa = str(aa)[2:-2] #name employee
                    mask = str(get_className(classIndex))
                    if (now < endCheckIn and now > startCheckIn):
                        # print("attendanced")
                        checkout = 'No'
                        attendance.loc[len(attendance)] = [date, Id, mask, timeStamp, checkout]
                    elif (now > startCheckOut and now < endCheckOut):
                        # print("checkouted")
                        id = attendance.index[attendance['Id'] == Id].tolist()
                        attendance.at[id, 'Checkout'] = 'Yes'


                else:
                    empInformation = {'Id': '', 'name': '', 'gender': '', 'dateOfBirth': '', 'position': '', 'mask': ''}
                #     # print("attendance failed ")
                #     # dont get name and id
                #     # cv2.rectangle(im, (x + 400, y - 150), (x + w + 400, y + h - 150), (255, 0, 0), 2)
                #     Id = '  Unknown  '
                #     tt = str(Id)
                #     confstr = "  {0}%".format(round(100 - conf))
                #
                #     # attendance failed
                #     cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (0, 0, 255), 2)
                #
                #     # hi???n th??? unknown
                #     cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

                # tt = str(tt)[2:-2]
            if (isFace == False):
                empInformation = {'Id': '', 'name': '', 'gender': '', 'dateOfBirth': '', 'position': '', 'mask': ''}

        attendance = attendance.sort_values(['Id', 'Mask'], ascending=[True, True])
        # cv2.imshow('Attendance', im)
        # open camera flask
        imgencode = cv2.imencode('.jpg', im)[1]
        strinData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + strinData + b'\r\n')

        # if () :
        if (now > endCheckOut and isExport == False):
            isExport = True
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour, Minute, Second = timeStamp.split(":")
            fileName = "Attendance" + os.sep + "Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
            attendance.to_csv(fileName, index=False)

            # print(fileName)
            with open(fileName) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                attendanceList = []

                for row in csv_reader:
                    if line_count == 0:
                        # print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        attendanceDetail = {
                            "date": row[0],
                            "userId": row[1],
                            "mask": row[2],
                            "checkIn": row[3],
                            "checkOut": row[4],
                        }
                        attendanceList.append(attendanceDetail)
                        line_count += 1
                # print(f'Processed {line_count} lines.')
            if (attendanceList != []):
                x = mycol.insert_many(attendanceList)

            print("Attendance Successful")

    cam.release()
    cv2.destroyAllWindows()


# Attendance Route Html
@app.route('/attendance')
def attendance():
    return render_template('attendance.html')


# show video Attendance
@app.route('/attendanceVideo')
def attendanceVideo():
    return Response(recognize_attendence(), mimetype='multipart/x-mixed-replace; boundary=frame')


# show video capture Image
@app.route('/captureImage')
def video():
    return Response(captureImage(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Index Route html
# @app.route('/')
# def main():
#     return render_template('index.html')

# Capture Image Route Html
@app.route('/create-employee')
def capture():
    if 'username' in session:
        return render_template('createEmployee.html', positions=positions)
    return redirect('/login')


# Check Id isNumber or Not
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


# function capture Image for training
def captureImage():
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder TrainingImage
            cv2.imwrite("TrainingImage" + os.sep + Id + '.' +
                        str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            # print(str(sampleNum))
            # display the frame
            # cv2.imshow('frame', img)
        # open camera flask
        imgencode = cv2.imencode('.jpg', img)[1]
        strinData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + strinData + b'\r\n')
        if sampleNum >= 100:
            break
    cam.release()
    cv2.destroyAllWindows()
    res = "Images Saved for ID : " + Id
    row = [Id]
    with open("EmployeeIds" + os.sep + "EmployeeIds.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


# Form to fill Id
IsCapture = False


@app.route('/capture-image', methods=['GET', 'POST'])
def captureVideo():
    if 'username' in session:
        global IsCapture
        IsCapture = True
        if request.method == 'POST':
            global Id
            # Id = request.form.get("Id")
            Id = '2021' + str(getNextSequence('activities')['seq'])
            # print('request.form.get("AA") ', request.form.get("AA"))
            employeeInfor = {
                'name': request.form.get('name'),
                'email': request.form.get('email'),
                'gender': request.form.get('gender'),
                'phoneNumber': request.form.get('phoneNumber'),
                'address': request.form.get('address'),
                'dateOfBirth': request.form.get('dateOfBirth'),
                'position': request.form.get('position'),
                # 'id': request.form.get('Id'),
                'id': Id,
            }

            existing_phoneNumber = listEmployeesCol.find_one({'phoneNumber': request.form['phoneNumber']})
            existing_email = listEmployeesCol.find_one({'email': request.form['email']})
            if (existing_phoneNumber != None or existing_email != None):
                return render_template('createEmployee.html', existingEmp=True, positions=positions)
            else:
                if is_number(Id):
                    listEmployeesCol.insert_one(employeeInfor)
                    return render_template('captureImage.html', empUpInfor=employeeInfor)
    return redirect('/login')


@app.route('/update-images/<id>')
def update_images(id):
    if 'username' in session:
        empUpId = {"id": id}
        empUp = listEmployeesCol.find(empUpId)
        for x in empUp:
            empUpInfor = x

        global Id
        Id = str(id)

        # you can use the the rowData from template
        if is_number(Id):
            return render_template('captureImage.html', empUpInfor=empUpInfor)
    return redirect('/login')


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[0])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# ----------- train images function ---------------
@app.route('/trainImage', methods=['GET', 'POST'])
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    Thread(target=recognizer.train(faces, np.array(Id))).start()
    # Below line is optional for a visual counter effect
    # Thread(target = counter_img("TrainingImage")).start()
    recognizer.write("TrainingImageLabel" + os.sep + "Trainner.yml")
    print("Trained All Images")
    return redirect('/')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        existing_username = admin.find_one({'name': request.form['username']})
        existing_useremail = admin.find_one({'email': request.form['email']})
        if (existing_username is None) and (existing_useremail is None) and (
                str(request.form['password']) == str(request.form['confirmPassword'])):
            hashpass = hashlib.md5(request.form['password'].encode('utf-8')).hexdigest()
            admin.insert({'name': request.form['username'], 'email': request.form['email'], 'password': hashpass})
            session['username'] = request.form['username']
            return redirect('/login')

        return render_template('register.html', registerFail=True)

    return render_template('register.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        login_user = admin.find_one({'name': request.form['username']})
        if login_user:
            if login_user['password'] == hashlib.md5(request.form['password'].encode('utf-8')).hexdigest():
                session['username'] = request.form['username']
                return redirect(url_for('list_employees'))
        return render_template('login.html', loginFail=True)

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.debug = True
    app.run(debug=True, host='localhost', port=9874)
