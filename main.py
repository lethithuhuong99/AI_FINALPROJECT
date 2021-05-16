import cv2
from flask import Flask , Response , render_template,request, redirect
import os
import datetime
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from threading import Thread

import pymongo
import csv


#Use Flask
app = Flask(__name__ , template_folder = 'html')

#Use Database with MongoDb
myclient = pymongo.MongoClient("mongodb+srv://admin:admin123@cluster0.9n8yb.mongodb.net/employees?retryWrites=true&w=majority")
mydb = myclient['employees']
mycol = mydb['attendancedbs']
listEmployeesCol = mydb['employeedbs']
countersCol = mydb['counters']

if(countersCol.count() ==0):
    countersCol.insert_one(
        {
            '_id': "activities",
            'seq': 0
        }
    )

def getNextSequence(name):
    ret = countersCol.find_and_modify(
        query={ '_id' : name },
        update={'$inc': {'seq': 1}},
        new=True
    )
    return ret

@app.route('/')
def list_employees():
    resultPois = listEmployeesCol.find()
    listEmp=[];
    for x in resultPois:
        listEmp.append(x)
    return render_template("listEmployees.html",len = len(listEmp), listEmp = listEmp)

@app.route('/list-attendance')
def list_attendance():
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
    return render_template("listAttendance.html",len = len(listAttendance), listAttendance = listAttendance)


@app.route('/update-employee/<id>')
def update_employee(id):
    print(id)
    empUpId = {"id": id}
    empUp = listEmployeesCol.find(empUpId)
    for x in empUp:
        empUpInfor = x

    # you can use the the rowData from template
    return render_template('updateEmployee.html', empUpInfor = empUpInfor)



@app.route('/updated-employee/<id>',  methods=['GET', 'POST'])
def updated_employee(id):
    if request.method == 'POST':
        print(id)
        employeeInfor = { "$set": {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'gender': request.form.get('gender'),
            'phoneNumber': request.form.get('phoneNumber'),
            'address': request.form.get('address'),
            'dateOfBirth': request.form.get('dateOfBirth'),
            'position': request.form.get('position'),
            'id':id,
        }}

        empUpId = {"id": id}

        listEmployeesCol.update_one(empUpId, employeeInfor)
    # empUp = listEmployeesCol.find(empUpId)
    # for x in empUp:
    #     empUpInfor = x

    # you can use the the rowData from template
    return redirect('/')

@app.route('/delete-employee/<id>')
def delete_employee(id):
    empDelId = {"id": id}
    listEmployeesCol.delete_one(empDelId)
    # you can use the the rowData from template
    return redirect('/')



#Load model Mask
model = load_model('MyTrainingModel.h5')
threshold=0.90


def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:
		return "No Mask"

#Attendance Function
def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
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

    while True:
        ret, im = cam.read()
        im = cv2.flip(im,1)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        now = datetime.datetime.now()
        hour = 19
        minute = 39
        startCheckIn = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        endCheckIn = now.replace(hour=hour, minute=minute, second=20, microsecond=0)
        startCheckOut = now.replace(hour=hour, minute=minute, second=30, microsecond=0)
        endCheckOut = now.replace(hour=hour, minute=minute, second=50, microsecond=0)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
        if((now < endCheckIn and now > startCheckIn) or (now > startCheckOut and now < endCheckOut)):
            for(x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

                crop_img = im[y:y + h, x:x + h]
                img = cv2.resize(crop_img, (32, 32))
                img = preprocessing(img)
                img = img.reshape(1, 32, 32, 1)
                prediction = model.predict(img)
                classIndex = model.predict_classes(img)
                probabilityValue = np.amax(prediction)

                if probabilityValue > threshold:
                    if classIndex == 0:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 255, 0), -2)
                        cv2.putText(im, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        print("Mask")
                    elif classIndex == 1:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (50, 50, 255), 2)
                        cv2.rectangle(im, (x, y - 40), (x + w, y), (50, 50, 255), -2)
                        cv2.putText(im, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        print("No Mask")
                if (100-conf) > 50:
                    # lấy tên và id
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # aa = df.loc[df['Id'] == Id]['Name'].values
                    confstr = "  {0}%".format(round(100 - conf))
                    tt = str(Id)

                    #xử lý điểm danh, lưu vào file
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    # aa = str(aa)[2:-2] #name employee
                    mask = str(get_className(classIndex))
                    if(now < endCheckIn and now > startCheckIn):
                        print("Da diem danh")
                        checkout = 'No'
                        attendance.loc[len(attendance)] = [ date, Id, mask , timeStamp, checkout ]
                    elif(now > startCheckOut and now < endCheckOut):
                        print("Da checkout")
                        id = attendance.index[attendance['Id'] == Id].tolist()
                        attendance.at[id,'Checkout'] = 'Yes'

                    # hiển thị điểm danh thành công
                    tt = tt + " [Pass]"
                    cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (0, 255, 0), 2)

                    # hiển thị tên người điểm danh
                    cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )

                else:
                    print("CHua diem danh")
                    # không lấy tên và id
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    Id = '  Unknown  '
                    tt = str(Id)
                    confstr = "  {0}%".format(round(100 - conf))

                    # điểm danh khong thành công
                    cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (0, 0, 255), 2)

                    # hiển thị unknown
                    cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

                tt = str(tt)[2:-2]

        attendance = attendance.sort_values(['Id', 'Mask'], ascending=[True,True])
        # cv2.imshow('Attendance', im)
        # open camera flask
        imgencode = cv2.imencode('.jpg', im)[1]
        strinData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + strinData + b'\r\n')

        # if () :
        if (now > endCheckOut):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
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
    if(attendanceList!=[]):
        x = mycol.insert_many(attendanceList)


    print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()

#Attendance Route Html
@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

#show video Attendance
@app.route('/attendanceVideo')
def attendanceVideo():
    return Response(recognize_attendence(),mimetype='multipart/x-mixed-replace; boundary=frame')

#show video capture Image
@app.route('/captureImage')
def video():
    return Response(captureImage(),mimetype='multipart/x-mixed-replace; boundary=frame')

#Index Route html
# @app.route('/')
# def main():
#     return render_template('index.html')

#Capture Image Route Html
@app.route('/create-employee')
def capture():
    return render_template('createEmployee.html')

#Check Id isNumber or Not
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


#function capture Image for training
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
                cv2.imwrite("TrainingImage" + os.sep + Id +'.' +
                            str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                print(str(sampleNum))
                # display the frame
                # cv2.imshow('frame', img)
            #open camera flask
            imgencode = cv2.imencode('.jpg', img)[1]
            strinData = imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + strinData + b'\r\n')
            if sampleNum > 50:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id
        row = [Id]
        with open("StudentDetails" + os.sep + "StudentDetails.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()


#Form to fill Id
IsCapture = False
@app.route('/capture-image',  methods=['GET', 'POST'])
def captureVideo():
    global IsCapture
    IsCapture = True
    if request.method == 'POST':
        global Id
        # Id = request.form.get("Id")
        Id = '2021'+ str(getNextSequence('activities')['seq'])
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
        listEmployeesCol.insert_one(employeeInfor)
        if is_number(Id):
            return render_template('captureImage.html',empUpInfor=employeeInfor)


@app.route('/update-images/<id>')
def update_images(id):
    print(id)
    empUpId = {"id": id}
    empUp = listEmployeesCol.find(empUpId)
    for x in empUp:
        empUpInfor = x

    global Id
    Id = str(id)

    # you can use the the rowData from template
    if is_number(Id):
        return render_template('captureImage.html', empUpInfor=empUpInfor)


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
@app.route('/trainImage', methods = ['GET', 'POST'])
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    Thread(target = recognizer.train(faces, np.array(Id))).start()
    # Below line is optional for a visual counter effect
    # Thread(target = counter_img("TrainingImage")).start()
    recognizer.write("TrainingImageLabel"+os.sep+"Trainner.yml")
    print("All Images")
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9874)