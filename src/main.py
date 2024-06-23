import cv2
import numpy as np
import mediapipe as mp

# Constants for object detection
thres = 0.3
nms_threshold = 0.3

# Load class names from file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Open webcam
cap = cv2.VideoCapture(0)

# Set the escape key variable
esc_key = 27

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and detect objects
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classIds[i] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('Object Recognition through Image Processing', cv2.flip(img, 1))

        if cv2.waitKey(1) & 0xFF == esc_key:
            break

cap.release()
cv2.destroyAllWindows()
