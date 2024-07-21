# Navigate to your project directory
cd C:\Users\ishan\source\repos\filter_application

# Remove the existing .git directory to start fresh
Remove-Item -Recurse -Force .git

# Initialize a new Git repository
git init

# Set the remote repository URL
git remote add origin https://github.com/rohanreddy1201/video-obj-filter-detection-application.git

# Create a .gitignore file
Set-Content .gitignore ".vs/"
Add-Content .gitignore "*.user"
Add-Content .gitignore "*.suo"
Add-Content .gitignore "*.cache"
Add-Content .gitignore "*.log"

# Create a README.md file
Set-Content README.md "# Interactive Video Filter Application

This project is an interactive video filter application with real-time object detection capabilities. Utilizing OpenCV for video processing and the YOLO (You Only Look Once) model for object detection, the application offers a user-friendly interface for applying various visual filters, detecting objects, and interacting with video feeds.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Future Opportunities](#future-opportunities)
- [Contributors](#contributors)
- [License](#license)

## Introduction

In the era of digital media, video content has become ubiquitous. Enhancing videos with visual effects and detecting objects within video feeds are increasingly important for applications in security, entertainment, and data analysis. This project aims to create an interactive application that combines real-time video filtering with object detection using machine learning techniques.

## Features

- **Real-Time Video Filtering**: Apply various visual filters in real-time, including Gaussian Blur, Median Filter, Bilateral Filter, Laplacian Edge Detection, and Color Space Conversions (RGB, HSV, LAB).
- **Object Detection**: Utilize the YOLOv3 model for real-time object detection, highlighting and labeling detected objects.
- **User-Friendly Interface**: Intuitive UI with buttons for selecting filters, a slider for adjusting filter intensity, and options for loading new videos and taking snapshots.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/rohanreddy1201/video-obj-filter-detection-application.git
   cd video-obj-filter-detection-application
   ```

2. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```sh
   python filter_application.py
   ```

2. **User Interface**:
   - Select various filters using the buttons.
   - Adjust filter intensity using the slider.
   - Load new videos and take snapshots through the provided options.

## Experiments

### Experiment 1: Real-Time Filter Application
- **Objective**: Assess the ability of the application to apply various filters in real-time without significant latency.
- **Results**: Successfully applied all supported filters in real-time with minimal latency.

### Experiment 2: Object Detection Accuracy
- **Objective**: Evaluate the accuracy and reliability of the YOLOv3 model for real-time object detection.
- **Results**: High accuracy in detecting and labeling objects in real-time with a low rate of false positives and missed detections.

### Experiment 3: User Interface Usability
- **Objective**: Test the usability and intuitiveness of the user interface.
- **Results**: Positive feedback from users with varying technical backgrounds.

## Future Opportunities

- **Advanced Object Detection Models**: Incorporate more advanced models like YOLOv4 or Faster R-CNN.
- **Additional Filters**: Expand the range of available filters and effects.
- **User Customization**: Allow users to upload custom filters and models.
- **Performance Optimization**: Improve the efficiency and speed of the application.
- **Cloud Integration**: Deploy the application on cloud platforms for handling larger videos and complex processing tasks.

## Contributors

- Rohan Reddy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [OpenCV Library](https://opencv.org/)
- [YOLO (You Only Look Once) Model](https://pjreddie.com/darknet/yolo/)
- [COCO Dataset](http://cocodataset.org/)
