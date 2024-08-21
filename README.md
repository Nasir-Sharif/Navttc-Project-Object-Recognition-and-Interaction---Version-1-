# Navttc-Project-Object-Recognition-and-Interaction---Version-1

# Object Recognition and Interaction System

## Overview
This project is an interactive object recognition system that utilizes the YOLOv5 model, trained on the COCO dataset, to detect and identify objects in real-time using a live camera feed. The system allows users to interactively click on detected objects, displaying the name of the selected object for educational and practical applications.

## Objective
The objective of this project is to:
- **Real-Time Object Detection:** Accurately identify and localize various objects in a video stream using the YOLOv5 model.
- **User Interaction:** Provide an interactive interface that allows users to click on detected objects and learn about them.
- **Educational and Practical Application:** Demonstrate the application of deep learning models in real-world scenarios.
- **Object Recognition and Analysis:** Analyze scenes for object classification, providing insights into the objects detected.

## Features
- **Real-Time Detection:** Utilizes a live camera feed for real-time object detection.
- **Interactive Interface:** Allows users to click on objects to see their names.
- **High Accuracy:** Leverages the YOLOv5 model, trained on the COCO dataset, for accurate detection.
- **Detailed Feedback:** Displays the confidence score and class of each detected object.

## Prerequisites
Before running the project, ensure that you have the following installed:
- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/installation/) package manager

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Nasir-Sharif/object-recognition-interaction.git
cd object-recognition-interaction
```

### 2. Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional Libraries
Make sure to have the following libraries installed:
```bash
pip install torch torchvision opencv-python
```

## YOLOv5 and COCO Dataset
### Why YOLOv5?
YOLOv5 is a state-of-the-art deep learning model for object detection, known for its speed and accuracy. It's ideal for real-time applications because it can process images quickly without sacrificing performance.

### Why COCO Dataset?
The COCO (Common Objects in Context) dataset is one of the most comprehensive datasets available for object detection, containing over 80 object categories. The YOLOv5 model trained on COCO is capable of recognizing a wide range of everyday objects with high accuracy.

## Code Explanation

### Object Detection and User Interaction
The code uses OpenCV to capture video from a camera, processes each frame using YOLOv5, and detects objects. Users can click on the detected objects to interact with them.

**Example Function: `click_event`**
```python
def click_event(event, x, y, flags, param):
    global selected_object
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in results.xyxy[0]:
            xmin, ymin, xmax, ymax, _, class_idx = i
            if xmin < x < xmax and ymin < y < ymax:
                selected_object = model.names[int(class_idx)]
                print(f'Selected Object: {selected_object}')
```

This function:
- Detects left mouse button clicks.
- Checks if the click is within any detected object's bounding box.
- Identifies and prints the name of the selected object.

### Main Script Workflow
1. **Loading the Model**: The YOLOv5 model is loaded using the `torch.hub.load()` function.
2. **Capturing Video**: OpenCV is used to capture video from the camera.
3. **Processing Frames**: Each frame is processed to detect objects, which are then displayed on the screen.
4. **User Interaction**: Users can click on detected objects to see their names and details.

## Running the Project
To run the object recognition and interaction system, use the following command:
```bash
python main.py
```

## Example Output
When the system detects objects, it will display bounding boxes around them. Clicking on an object will print its name in the console.

## Conclusion
This project demonstrates the power of deep learning in real-time object detection and provides an interactive platform for users to engage with and learn about detected objects. By leveraging the YOLOv5 model and COCO dataset, the system offers both high accuracy and practical usability.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- **YOLOv5**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- **COCO Dataset**: [COCO - Common Objects in Context](https://cocodataset.org/)
```
