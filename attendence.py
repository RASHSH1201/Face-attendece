import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load images and extract class names
path = 'img'
images = []
classnames = []
mylist = os.listdir(path)

print("Images found:", mylist)

for cls in mylist:
    curimg = cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    classnames.append(os.path.splitext(cls)[0])

print("Class names:", classnames)

# Function to encode known faces
def findEncoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodelist.append(encode)
        except IndexError:
            print("No face detected in one of the images, skipping...")
    return encodelist

# Function to mark attendance
def markattendence(name):
    with open('attendence.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = [line.split(',')[0] for line in mydatalist]
        if name not in namelist:
            now = datetime.now()
            dstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dstring}')
            print(f"Attendance marked for {name}")

# Encode known faces
encodelistknown = findEncoding(images)
print("Encoding Complete")

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facescurrframe = face_recognition.face_locations(imgS)
    encodecurrframe = face_recognition.face_encodings(imgS, facescurrframe)

    for encodeface, faceloc in zip(encodecurrframe, facescurrframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        faceDis = face_recognition.face_distance(encodelistknown, encodeface)
        matchindex = np.argmin(faceDis)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(f"Detected: {name}")
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            markattendence(name)

    cv2.imshow('Webcam', img)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLoctest = face_recognition.face_locations(imgtest)[0]
# encodetest = face_recognition.face_encodings(imgtest)[0]
# cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodeElon],encodetest)
# faceDis = face_recognition.face_distance([encodeElon],encodetest)