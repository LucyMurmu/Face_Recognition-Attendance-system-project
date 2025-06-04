import os
import cv2
path="C:/Users/mishr/Downloads/Face-recognition-Attendance-System-Project-main/Face-recognition-Attendance-System-Project-main/Image"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load images safely
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Error loading image: {cl}")
        continue
    images.append(curImg)