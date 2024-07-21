import cv2
import numpy as np
from tkinter import Tk, filedialog
from color_transfer import convert_color_space_RGB_to_Lab, convert_color_space_Lab_to_RGB, convert_color_space_BGR_to_RGB, convert_color_space_RGB_to_BGR  # Import the color transfer functions

# Initialize global variables
filter_type = "None"
filter_intensity = 1
snapshot_count = 0
target_image = None  # Target image for color transfer
object_detection_enabled = False

# Paths to YOLO model files
model_weights = 'yolov3.weights'
model_cfg = 'yolov3.cfg'
model_names = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open(model_names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Function to apply selected filter
def apply_filter(image, filter_type, intensity):
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (intensity * 2 + 1, intensity * 2 + 1), 0)
    elif filter_type == "Median Filter":
        return cv2.medianBlur(image, intensity * 2 + 1)
    elif filter_type == "Bilateral Filter":
        return cv2.bilateralFilter(image, intensity * 2 + 1, 75, 75)
    elif filter_type == "Laplacian":
        return cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F, ksize=intensity * 2 + 1))
    elif filter_type == "RGB Color Space":
        return convert_color_space_BGR_to_RGB(image)
    elif filter_type == "HSV Color Space":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif filter_type == "LAB Color Space":
        rgb_image = convert_color_space_BGR_to_RGB(image)
        lab_image = convert_color_space_RGB_to_Lab(rgb_image)
        return convert_color_space_RGB_to_BGR(convert_color_space_Lab_to_RGB(lab_image))
    else:
        return image

# Function to update filter intensity
def update_intensity(val):
    global filter_intensity
    filter_intensity = max(1, val)

# Function to handle button events
def select_filter(filter_name):
    global filter_type, target_image, object_detection_enabled
    if filter_name == 'Object Detection':
        object_detection_enabled = not object_detection_enabled
    else:
        filter_type = filter_name
        object_detection_enabled = False

# Function to perform object detection
def object_detection(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Function to resize frame to fit within the window
def resize_frame(frame, max_width, max_height):
    height, width = frame.shape[:2]
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)

    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return cv2.resize(frame, (width, height))

# Function to take a snapshot
def take_snapshot():
    global snapshot_count, filtered_frame
    snapshot_count += 1
    filename = f'snapshot_{snapshot_count}.png'
    cv2.imwrite(filename, filtered_frame)
    print(f'Snapshot saved as {filename}')

# Draw buttons for filter selection and snapshot
def draw_buttons():
    buttons_img = np.zeros((400, 800, 3), np.uint8)
    buttons_img[:] = (255, 255, 255)

    button_texts_row1 = ['Gaussian Blur', 'Median Filter', 'Bilateral Filter', 'Laplacian']
    button_texts_row2 = ['RGB Color Space', 'HSV Color Space', 'LAB Color Space', 'Snapshot']
    button_texts_row3 = ['Load New Video', 'Object Detection']

    x_positions_row1 = [20, 200, 380, 560]
    x_positions_row2 = [20, 220, 420, 620]
    x_positions_row3 = [20, 220]

    y_position_row1 = 50
    y_position_row2 = 150
    y_position_row3 = 250

    for text, x in zip(button_texts_row1, x_positions_row1):
        cv2.putText(buttons_img, text, (x, y_position_row1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for text, x in zip(button_texts_row2, x_positions_row2):
        cv2.putText(buttons_img, text, (x, y_position_row2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for text, x in zip(button_texts_row3, x_positions_row3):
        cv2.putText(buttons_img, text, (x, y_position_row3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return buttons_img

def button_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 < x < 170 and 30 < y < 70:
            select_filter('Gaussian Blur')
        elif 200 < x < 350 and 30 < y < 70:
            select_filter('Median Filter')
        elif 380 < x < 530 and 30 < y < 70:
            select_filter('Bilateral Filter')
        elif 560 < x < 710 and 30 < y < 70:
            select_filter('Laplacian')
        elif 20 < x < 200 and 130 < y < 170:
            select_filter('RGB Color Space')
        elif 220 < x < 400 and 130 < y < 170:
            select_filter('HSV Color Space')
        elif 420 < x < 600 and 130 < y < 170:
            select_filter('LAB Color Space')
        elif 620 < x < 800 and 130 < y < 170:
            take_snapshot()
        elif 20 < x < 200 and 230 < y < 270:
            load_new_video()
        elif 220 < x < 400 and 230 < y < 270:
            select_filter('Object Detection')

# Load video file using file dialog
def load_video():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

def load_new_video():
    global capture, video_path
    video_path = load_video()
    capture.release()
    capture = cv2.VideoCapture(video_path)

# Setup UI
cv2.namedWindow('Control Video Feed')
cv2.namedWindow('Filtered Video Feed')
cv2.namedWindow('Controls')
cv2.createTrackbar('Intensity', 'Controls', 1, 40, update_intensity)

# Main loop
max_width = 800
max_height = 600
video_path = load_video()
capture = cv2.VideoCapture(video_path)

while True:
    ret, frame = capture.read()
    if not ret:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_frame = frame.copy()  # Save the current frame for color transfer target
    filtered_frame = apply_filter(frame, filter_type, filter_intensity)
    
    if object_detection_enabled:
        filtered_frame = object_detection(filtered_frame)
    
    resized_frame = resize_frame(frame, max_width, max_height)
    resized_filtered_frame = resize_frame(filtered_frame, max_width, max_height)
    
    buttons_img = draw_buttons()
    cv2.imshow('Controls', buttons_img)
    cv2.imshow('Control Video Feed', resized_frame)
    cv2.imshow('Filtered Video Feed', resized_filtered_frame)
    cv2.setMouseCallback('Controls', button_click)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break

capture.release()
cv2.destroyAllWindows()
