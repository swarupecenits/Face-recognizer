# Face Recognition Model using OpenCV and Python

This project demonstrates a simple face recognition model using OpenCV and Python. The model encodes faces from images stored in a directory and detects known faces in real-time from a webcam feed.

## Features

- Encodes faces from images in a specified directory.
- Detects known faces in real-time using a webcam.
- Displays the name of the detected person based on the folder name containing their images.

## Requirements

- Python 3.6+
- OpenCV
- face_recognition
- numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-opencv-python.git
   cd face-recognition-opencv-python


Install the required packages:

bash

    pip install opencv-python face_recognition numpy

Directory Structure

The project directory should have the following structure:



face-recognition-opencv-python/
│
├── images/
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── person2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── simple_facerec.py
├── main.py
└── README.md


Usage

    Place your images in the images directory. Create a subdirectory for each person and place their images inside it.

    Run the main script:

    bash

    python main.py

    The script will start the webcam and display the video feed with recognized faces labeled.

Code Explanation
simple_facerec.py

This file contains the SimpleFacerec class, which handles encoding and face recognition.

    load_encoding_images(self, images_path): Loads and encodes images from the specified directory.
    detect_known_faces(self, frame): Detects known faces in the given frame.
    get_encoded_image_count(self): Returns the count of encoded images.

main.py

This file contains the main script that uses the SimpleFacerec class to perform face recognition in real-time using a webcam.
Example

    Add images of "John Doe" and "Jane Smith" in their respective folders inside the images directory:



images/
├── JohnDoe/
│   ├── john1.jpg
│   ├── john2.jpg
│   └── ...
└── JaneSmith/
    ├── jane1.jpg
    ├── jane2.jpg
    └── ...


Run the script:


python main.py

The webcam feed will open and display the names "JohnDoe" and "JaneSmith" when their faces are detected.