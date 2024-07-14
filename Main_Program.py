 # Header Files are are used and Imported.
from my_CNN_model import *
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Load the model built in the previous step
my_model = load_my_CNN_model("my_model")
# detect faces.
face_cascade = cv2.CascadeClassifier("FaceCoordinates/frontalface.xml")
# Define the upper and lower boundaries to be in "Blue" Colour.
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])
# Define a 5x5 kernel.
kernel = np.ones((5, 5), np.uint8)
# Define filters according the the images of the available products.
filters = [
    "images/sunglasses.png",
    "images/sunglasses_2.png",
    "images/sunglasses_3.jpg",
    "images/sunglasses_4.png",
    "images/sunglasses_5.jpg",
    "images/sunglasses_6.png",
    "images/sunglasses_7.png",
    "images/sunglasses_8.png",
    "images/sunglasses_9.png",
]
filterIndex = 0
# Video Loading.
camera = cv2.VideoCapture(0)
# Variables Initialize.
last_click_time = time.time()
total_frames = 0
accurate_frames = 0
accuracy_data = []
# Click Button to Handle to change the glasses.
def handle_button_click(event, x, y, flags, param):
    global filterIndex, last_click_time
    #Conditions.
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the 'Next Filter' button was clicked
        if 500 <= x <= 620 and 10 <= y <= 65:
            # Check if it's been more than 0.2 seconds since the last click
            if time.time() - last_click_time > 0.2:
                filterIndex = (filterIndex + 1) % 6
                last_click_time = time.time()
        # Check if the 'Exit' button was clicked
        elif 10 <= x <= 100 and 10 <= y <= 65:
            cv2.destroyAllWindows()
            exit()
cv2.namedWindow("Selfie Filters")
cv2.setMouseCallback("Selfie Filters", handle_button_click)
# Frame Settings and parameters.
while True:
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Add the 'Next Filter' button to the frame
    frame = cv2.rectangle(frame, (500, 10), (620, 65), (235, 50, 50), -1)
    cv2.putText(
        frame,
        "NEXT FILTER",
        (512, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
# Add the 'Exit' button to the frame
    frame = cv2.rectangle(frame, (10, 10), (100, 65), (235, 50, 50), -1)
    cv2.putText(
        frame,
        "EXIT",
        (30, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
#Facial Checks.
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    cnts, _ = cv2.findContours(
        blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    center = None
# if condition check the facial structure and enclosed into circle.
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
# For loop is used to check all the facial points view are possiable on the face.
    for x, y, w, h in faces:
        gray_face = gray[y : y + h, x : x + w]
        color_face = frame[y : y + h, x : x + w]
        # Normalized face for Gray.
        gray_normalized = gray_face / 255
        original_shape = gray_face.shape
        face_resized = cv2.resize(
            gray_normalized, (96, 96), interpolation=cv2.INTER_AREA
        )
        face_resized = face_resized.reshape(1, 96, 96, 1)
# KeyPoints Generation for the Predicted face size.
        keypoints = my_model.predict(face_resized)
        keypoints = keypoints * 48 + 48
# Resize the Recored faceSize. 
        face_resized_color = cv2.resize(
            color_face, (96, 96), interpolation=cv2.INTER_AREA
        )
        # coordiante points.
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))
        # Modifying and optmizing the facial size of the gunglasses according to the user face.
        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
        sunglass_width = int((points[7][0] - points[9][0]) * 1.1)
        sunglass_height = int((points[10][1] - points[8][1]) / 1.1)
        sunglass_resized = cv2.resize(
            sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_CUBIC
        )
        transparent_region = sunglass_resized[:, :, :3] != 0
        face_resized_color[
            int(points[9][1]) : int(points[9][1]) + sunglass_height,
            int(points[9][0]) : int(points[9][0]) + sunglass_width,
            :,
        ][transparent_region] = sunglass_resized[:, :, :3][transparent_region]

        frame[y : y + h, x : x + w] = cv2.resize(
            face_resized_color, original_shape, interpolation=cv2.INTER_CUBIC
        )
    cv2.imshow("Selfie Filters", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()