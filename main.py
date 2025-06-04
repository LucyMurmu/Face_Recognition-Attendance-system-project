import cv2
import numpy as np
import face_recognition
import time
import sys
import os

# Image paths
img_path = "C:/Users/mishr/Downloads/Face-recognition-Attendance-System-Project-main/Face-recognition-Attendance-System-Project-main/Image/ankitesh.jpg"

# Check if image files exist
if not os.path.exists(img_path):
    print(f"Error: Image file '{img_path}' not found.")
    sys.exit()

# Load images
imgModi = face_recognition.load_image_file(img_path)
imgTest = face_recognition.load_image_file(img_path)

# Convert images to RGB
imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect face locations
faceloc = face_recognition.face_locations(imgModi)
facelocTest = face_recognition.face_locations(imgTest)

# Ensure faces are detected
if not faceloc or not facelocTest:
    print("Error: Face not detected in one or both images.")
    sys.exit()

faceloc = faceloc[0]
facelocTest = facelocTest[0]

# Encode faces
encodeModi = face_recognition.face_encodings(imgModi)
encodeTest = face_recognition.face_encodings(imgTest)

if not encodeModi or not encodeTest:
    print("Error: Could not encode one or both faces.")
    sys.exit()

encodeModi = encodeModi[0]
encodeTest = encodeTest[0]

# Draw rectangles around detected faces
cv2.rectangle(imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

# Compare faces
results = face_recognition.compare_faces([encodeModi], encodeTest)
faceDis = face_recognition.face_distance([encodeModi], encodeTest)

print(f"Match: {results[0]}, Distance: {round(faceDis[0], 2)}")

# Display match result on the test image
cv2.putText(imgTest, f'Match: {results[0]} {round(faceDis[0], 2)}', (50, 50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

# Show images
cv2.imshow('Original Image', imgModi)
cv2.imshow('Test Image', imgTest)

# Auto close after 1 minute (60000 milliseconds)
cv2.waitKey(5000)
cv2.destroyAllWindows()

print("Time's up! Exiting program.")
sys.exit()
