import cv2
import numpy as np
import face_recognition
import os
import time
import pandas as pd
from datetime import datetime
import smtplib
from email.message import EmailMessage

# ===== EMAIL CONFIGURATION =====
EMAIL_ADDRESS = "meetankitesh420@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "cxsn sxps pmpy kpqw"    # Replace with your App Password

# Mapping names to their email addresses
email_mapping = {
    "ANKITESH": "meetankitesh420@gmail.com",
    "BINAY": "binaykumarsaw70@gmail.com",
    "LALAN": "mrlalankumar282@gmail.com",
    "Lucy": "btech10543.22@bitmesra.ac.in",
}

# ===== EMAIL FUNCTION =====
def send_email(to_email, subject, body):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg.set_content(body)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")

# ===== LOAD IMAGES AND ENCODINGS =====
path = "C:/Users/mishr/OneDrive/Desktop/Face-recognition-Attendance-System-Project-main/Face-recognition-Attendance-System-Project-main/Image"
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    if curImg is None:
        print(f"Error loading image: {cl}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0].upper())

print(f"Loaded Class Names: {classNames}")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
        else:
            print("Warning: No face found in an image.")
    return encodeList

# ===== ATTENDANCE TRACKING =====
attendance_data = {}
file_path = "Attendance.xlsx"

def markAttendance(name, status="Entry"):
    time_now = datetime.now()
    tString = time_now.strftime('%H:%M:%S')
    dString = time_now.strftime('%d/%m/%Y')

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time", "Duration (sec)", "Date"])
        df.to_excel(file_path, index=False)

    df = pd.read_excel(file_path)

    if status == "Entry":
        if name not in attendance_data:
            attendance_data[name] = time.time()
            new_entry = pd.DataFrame([[name, tString, "", "", dString]], columns=df.columns)
            df = pd.concat([df, new_entry], ignore_index=True)
            print(f"Entry marked for {name} at {tString} on {dString}")

            if name in email_mapping:
                subject = "Attendance Notification - Present"
                body = f"Hello {name},\n\nYou have been marked PRESENT at {tString} on {dString}.\n\nRegards,\nAttendance System"
                send_email(email_mapping[name], subject, body)

    elif status == "Exit" and name in attendance_data:
        entry_time = attendance_data.pop(name)
        duration = int(time.time() - entry_time)

        for i in range(len(df)):
            if df.loc[i, "Name"] == name and pd.isna(df.loc[i, "Exit Time"]):
                df.at[i, "Exit Time"] = tString
                df.at[i, "Duration (sec)"] = duration
                break

        print(f"Exit marked for {name} at {tString} (Total time: {duration} sec)")

        if name in email_mapping:
            subject = "Attendance Notification - Exit"
            body = f"Hello {name},\n\nYou have been marked EXIT at {tString} on {dString}.\nYou spent {duration} seconds.\n\nRegards,\nAttendance System"
            send_email(email_mapping[name], subject, body)

    df.to_excel(file_path, index=False)

# ===== START CAMERA & PROCESS FRAMES =====
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected!")
    exit()

start_time = time.time()
last_seen = {}
time_threshold = 5  # seconds

def process_frame():
    success, img = cap.read()
    if not success:
        print("Error: Couldn't capture frame.")
        return None

    imgS = cv2.resize(img, (320, 240))
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    detected_names = []

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex] and faceDis[matchIndex] < 0.5:
                name = classNames[matchIndex]
                detected_names.append(name)

                y1, x2, y2, x1 = [coord * 2 for coord in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name, "Entry")
                last_seen[name] = time.time()

    for name, last_time in list(last_seen.items()):
        if name not in detected_names and time.time() - last_time > time_threshold:
            markAttendance(name, "Exit")
            del last_seen[name]

    return img

while True:
    frame = process_frame()
    if frame is None:
        break

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing program.")
        break

    if time.time() - start_time > 30:
        print("Auto-closing after 30 seconds.")
        break

cap.release()
cv2.destroyAllWindows()
