# Object Recognition through Image Processing with Webcam

This project demonstrates real-time object detection using a webcam feed. It utilizes OpenCV for image processing and object detection, MediaPipe for face detection, and pre-trained deep learning models.

## Features

- Real-time object detection using a webcam.
- Integration with MediaPipe for enhanced face detection.
- Uses SSD MobileNet V3 model pre-trained on COCO dataset.
- Draws bounding boxes and labels around detected objects.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- MediaPipe

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/object-recognition-webcam.git
   cd object-recognition-webcam
   ```

2. Install the required Python packages:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. Download the required files and place them in the project directory:
   - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
   - [ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt)
   - [frozen_inference_graph.pb](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz)

## Usage

1. Run the `main.py` script:
   ```bash
   python main.py
   ```

2. The script will open the webcam and start detecting objects in real-time. Press the 'Esc' key to exit the application.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


