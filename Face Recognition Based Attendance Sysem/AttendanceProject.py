import cv2  #cv2 is opencv library designed to process images and videos to identify  objects,faces etc
import numpy as np #numpy(numerical python) is a library for working with arrays,has many functions to work using  mathecical opreations it provides array object which is much more faster than list
import face_recognition #python library for face recognition and manipulation
import os
import keyboard
from datetime import datetime

path ='images'
images =[]                  #images is a list to store the actual images for the project.
classNames =[]              #classnames will store the names of person whoose images are present inside  the images directory.
myList=os.listdir(path)     #listing all the files of path directory and storing them in myList array

for cl in myList:           #we are moving through each images inside myList and storin them in 'images' array too
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]                                   #encodeList will store the encodings of the each face using FaceReco library.
    for img in images:
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    #converting each image's color to RGB format from BGR
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDatalist = f.readlines()
        nameList=[]
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now= datetime.now()
            dtString = now.strftime('%d-%m-%y')
            dtString2 = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString},{dtString2}')



encodeListKnown  = findEncodings(images)
print(len(encodeListKnown))
print('Encodings Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)              #stores the locations points in the current face in front of webcam
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)   #stores the encodings of the current face in front of webcam.

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) #returns true or false value and stores in 'matches' list
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #stores distance values in same order to the numpy array
        #print(faceDis)
        matchIndex = np.argmin(faceDis)       # here 'matchIndex' is storing the lowest value inside the array using Numpy.

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
         #   print(name)

          #now its a code for creating a rectangular box over the realtime screen.
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,"press Q to quit",(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)     #imshow function is adding image overlay over Webcam
    cv2.waitKey(1)               #here we are processing image after wait of each 1ms.
    if keyboard.is_pressed('q'):
        break









