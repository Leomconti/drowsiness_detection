import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import dlib
import cv2
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import playsound
import argparse
import time

alarm_sound = "drowsiness_detection/resources/alarm.wav"


def sound_alarm(alarm_sound_path):  # Play the alarm sound
    playsound.playsound(alarm_sound_path)

# Calculate the eye aspect ratio (EAR)


# Calculate the euclidean distances between the two sets of vertical points (x, y)-coordinates
# Input will be the left eye and right eye indexes
def calculate_ear(eye):

    V1 = dist.euclidean(eye[1], eye[5])
    V2 = dist.euclidean(eye[2], eye[4])

    # Calculate the euclidean distance between the horizontal coordinates
    H = dist.euclidean(eye[0], eye[3])

    # Calculate the eye aspect ratio
    ear = (V1 + V2) / (2.0 * H)

    return ear


def process_facial_landmarks(frame):
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    
    ear = 0
    
    for (i, rect) in enumerate(rects):
        reference_points = predictor(gray, rect)
        reference_points = face_utils.shape_to_np(reference_points)
        
        # Extract the eyes coodinates based on the first and last points of each
        left_eye = reference_points[left_eye_start:left_eye_end]
        right_eye = reference_points[right_eye_start:right_eye_end]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate the convex hull for left and right eyes
        # Draw the contours to show in the video stream
        left_eye_hull= cv2.convexHull(left_eye)
        right_eye_hull= cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)

    return frame, ear


# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("drowsiness_detection/resources/shape_predictor_68_face_landmarks.dat")



# Define constants for the EAR and the number of consecutive frames the eye must be below the threshold
EAR_THRESHOLD = 0.25
QNT_CONSECUTIVE_FRAMES = 25

# Initialize the frame counter and the alarm boolean
COUNTER = 0
ALARM_ON = False

# Initialize the predictor and detecor
print("🤖 Loading facial landmarks predictor...")
# Using dlib HOG feature descriptor
detector = dlib.get_frontal_face_detector()
predicor = dlib.shape_predictor("drowsiness_detection/resources/shape_predictor_68_face_landmarks.dat")

# Grab the right and left eyes indexes


# Arguument parser for which webcam to use and if the alarm should be played
# ap = argparse.ArgumentParser()
# ap.add_argument("-a", "--alarm", type=int, default="",
#                 help="Use the sound alarm (0: No / 1: Yes)")
# ap.add_argument("-w", "--webcam", type=int, default=0,
#                 help="Index of the webcam on system")

# webcam_index = ap.parse_args().webcam
# alarm = ap.parse_args().alarm

webcam_index = 0
alarm = 1

# Read webcam video using imutils video stream
# It will load the video faster, although not the best quality
video_stream = VideoStream(src=webcam_index).start()
time.sleep(0.2)

# Start the matplotlib graph that will show the ear calculations
y = [None] * 100
x = np.arange(0, 100)
fig = plt.figure()
ax = fig.add_subplot(111)
lines, = ax.plot(x, y)

# Read and process frames from the webcam
while True:
    # Read the frame from the webcam, resize it, and convert it to grayscale
    frame = video_stream.read()
    frame = imutils.resize(frame, width=700)

    frame, ear = process_facial_landmarks(frame)
    
    # y is initialized with 100 Nones, and we always need 100 elements
    # So the first one is removed and the ear is appended
    y.pop(0)
    y.append(ear)
    
    # Update the graph
    plt.xlim([0, 100])
    plt.ylim([0, 0.4])
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    plt.show(block=False)
    
    # Define the data and draw
    lines.set_ydata(y)
    fig.canvas.draw()
    
    time.sleep(0.01)
    
    # Check if EAR is below the defined threshold
    # Verify if the alarm should sound
    if ear < EAR_THRESHOLD:
        COUNTER += 1
    
        if COUNTER >= QNT_CONSECUTIVE_FRAMES:
            
            if not ALARM_ON:
                ALARM_ON = True

                if alarm == 1:
                    thread = Thread(target=sound_alarm, args=(alarm_sound))
                    thread.start()
        
        cv2.putText(frame, "[ALERT] DROWSINESS!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
    else:
        COUNTER = 0
        ALARM_ON = False


            
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Break the loop if 'q' is pressed
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the video stream and close the window
cv2.destroyAllWindows()
video_stream.stop()
