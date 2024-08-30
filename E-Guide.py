# Import packages # last onee
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pyttsx3
import speech_recognition as sr
import pygame
from pygame import mixer
import webcolors
from scipy.spatial import KDTree

# Function to get color name from RGB value using KDTree for closest match
def get_color_name(rgb_tuple):
    css3_db = webcolors.CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []

    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]



def mySpeak(message):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say('{}'.format(message))
    engine.runAndWait()
mySpeak('Hello')
mySpeak('Your Journey Is Started')

# Define constants for beep files
BEEP_FAST = "/home/pi/Desktop/beeps_fast.wav"
BEEP_MEDIUM = "/home/pi/Desktop//beeps_medium.wav"
BEEP_SLOW = "/home/pi/Desktop/beeps_slow.wav"


def detect_object_by_voice():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        mySpeak("Please say the object you want to detect")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        object_name = recognizer.recognize_google(audio)
        print(f"You said: {object_name}")
        mySpeak(f"Searching for {object_name}")
        return object_name.lower()  # Convert to lowercase for consistency
    except sr.UnknownValueError:
        print("Could not understand audio")
        mySpeak("Sorry, I could not understand. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        mySpeak("Sorry, I'm having trouble processing your request.")
        return None

def get_direction(x, width):
    if x < width / 12:
        return "1 o'clock"
    elif x < width / 6:
        return "2 o'clock"
    elif x < width / 4:
        return "3 o'clock"
    elif x < width / 3:
        return "4 o'clock"
    elif x < 5 * width / 12:
        return "5 o'clock"
    elif x < width / 2:
        return "6 o'clock"
    elif x < 7 * width / 12:
        return "7 o'clock"
    elif x < 2 * width / 3:
        return "8 o'clock"
    elif x < 3 * width / 4:
        return "9 o'clock"
    elif x < 5 * width / 6:
        return "10 o'clock"
    elif x < 11 * width / 12:
        return "11 o'clock"
    else:
        return "12 o'clock"

def get_distance(y, height):
    if y < height / 3:
        return "near"
    elif y > 2 * height / 3:
        return "far"
    else:
        return "medium"




# Function to get verbosity level from voice
def get_verbosity_level():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening for verbosity level...")
        mySpeak("Please say the verbosity level. Options are: basic or detailed.")
        recognizer.adjust_for_ambient_noise(source)
        aud2 = recognizer.listen(source)

    try:
        verbosity = recognizer.recognize_google(aud2)
        print(f"You said: {verbosity}")
        mySpeak(f"You said {verbosity}")
        if verbosity.lower() in ["basic", "detailed", "detail", "details"]:
            return verbosity.lower()
        else:
            mySpeak("Invalid option. Please say either basic or detailed.")
            return get_verbosity_level()
    except sr.UnknownValueError:
        print("Could not understand audio")
        mySpeak("Sorry, I could not understand the audio.")
        return get_verbosity_level()
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        mySpeak(f"Sorry, I could not request results; {e}")
        return get_verbosity_level()


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Initialize Pygame for audio feedback
pygame.mixer.init()

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

#frame_rate_calc = 1
#freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(imW, imH), framerate=15).start()
time.sleep(1)


while True:
    object_to_detect = detect_object_by_voice()
    if object_to_detect is None:
        continue  # Retry if no valid object name detected
        
    verbosity_level = get_verbosity_level()
    if verbosity_level is None:
        continue  # Retry if no valid verbosity level detected

    start_time = time.time()
    detected_object = False

    while time.time() - start_time < 60:  # Search for the object for 1 minute
        #t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                object_id = int(classes[i])
                if labels[object_id].lower() == object_to_detect:
                    detected_object = True

                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    object_name = labels[object_id]
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    #message = f"{object_name} at {get_direction((xmin + xmax) / 2, imW)} and {get_distance((ymin + ymax) / 2, imH)}"
                    #mySpeak(message)

                    direction = get_direction((xmin + xmax) / 2, imW)
                    distance = get_distance((ymin + ymax) / 2, imH)

                    if verbosity_level == "basic":
                        print(f"Detected {object_name} at {direction}")
                        mySpeak(f"{object_name} at {direction}")
                    else:  # detailed verbosity
                        color = frame[(ymin+ymax)//2, (xmin+xmax)//2]
                        color_name = get_color_name(color)  # Get color name
                        print(f"Detected {object_name} at {direction} with color {color_name}")
                        mySpeak(f"{object_name} at {direction} with color {color_name}")

                    beep_file = {
                       "near": BEEP_FAST,
                        "medium": BEEP_MEDIUM,
                        "far": BEEP_SLOW
                        }.get(distance, BEEP_SLOW)  # default to slow beep if distance is unknown
                    pygame.mixer.music.load(beep_file)
                    pygame.mixer.music.play()
                    #time.sleep(0.5)  # wait for the beep to finish
                    #pygame.mixer.music.stop()
                    #pygame.quit()

                    time.sleep(0.5)
                    pygame.mixer.music.stop()

                    if (ymin + ymax) / 2 > (2 * imH) / 3:
                        print(f"{object_name} is close")
                        mySpeak(f"{object_name} is close")
                        object_to_detect = detect_object_by_voice()  # Ask again for the object
                        verbosity_level = get_verbosity_level()  # Ask again for verbosity level

                    # Add a small delay to avoid overloading the CPU
                    time.sleep(0.2)


                

    if not detected_object:
        mySpeak(f"No {object_to_detect} detected")

    cv2.imshow('Object detector', frame)
    #t2 = cv2.getTickCount()
    #time1 = (t2 - t1) / freq
    #frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

videostream.stop()
cv2.destroyAllWindows()

