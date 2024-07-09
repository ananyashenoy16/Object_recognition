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

print("Select an option:")
print("1. Recognize objects through a photo")
print("2. Recognize objects through a camera")

option = int(input("Enter your choice: "))

if option == 1:
    # Recognize objects through a photo
    img_path = input("Enter the path to the image: ")
    img = cv2.imread(img_path)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    # print("classNames:", classNames)
    # print("classIds:", classIds)
    # print("Length of classNames:", len(classNames))
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[(classIds[i] - 1) % len(classNames)].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Recognition through Image Processing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif option == 2:
    # Recognize objects through a camera
    cap = cv2.VideoCapture(0)

    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
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
                cv2.putText(img, classNames[(classIds[i] - 1) % len(classNames)].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # Flip the image horizontally for a selfie-view display
            cv2.imshow('Object Recognition through Image Processing', cv2.flip(img, 1))

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid option. Please try again.")