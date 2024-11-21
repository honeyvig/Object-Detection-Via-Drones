# Object-Detection-Via-Drones
To build a project using computer vision techniques for object detection via drones and display the results on a mobile phone, the general process will consist of several key steps:

    Drone Image Capture: The drone will capture real-time images or videos of the scene using its onboard camera.
    Object Detection: The captured images will be processed to detect objects of interest using computer vision techniques.
    Statistics Extraction: Once objects are detected, we will extract specific statistics such as the number of detected objects, their size, position, and other relevant metrics.
    User Interface (Mobile): The processed data and images will be sent to a mobile application where the user can view the results and interact with the system.

Key Components and Technologies:

    Computer Vision (CV): Use pre-trained models like YOLO (You Only Look Once), MobileNet, or Detectron2 for object detection.
    Drone Communication: Use a protocol like DJI SDK (for DJI drones) or PX4 for communicating with the drone and getting video feed.
    Mobile UI: Use frameworks like Flutter or React Native for building a mobile app to display results.
    Cloud/Local Processing: Optionally, you can run object detection on the drone directly or stream data to a server/cloud for processing.

Below is a Python script that illustrates how you might implement such a system. This will primarily focus on the object detection part using a drone camera (for example, via streaming) and sending the results to a mobile UI.
Python Code Example

    Drone Stream Handling: Capture video feed from the drone.
    Object Detection Using OpenCV and YOLO: Process each frame and detect objects.
    Sending Results to Mobile UI: Stream data to a mobile app using a web server (Flask or FastAPI).

1. Drone Video Feed and Object Detection (YOLO)

The first step is to set up object detection on the images streamed by the drone.

import cv2
import numpy as np

# Load YOLO pre-trained weights and configuration
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Process video feed from the drone camera
def process_video_stream(video_stream_url):
    cap = cv2.VideoCapture(video_stream_url)
    net, output_layers = load_yolo_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Extract object information
        boxes, confidences, class_ids = [], [], []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Set confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maxima suppression to remove duplicates
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the frame with bounding boxes
        cv2.imshow("Drone Object Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example Usage
video_stream_url = 'http://192.168.10.1:8080/video'  # Example URL for drone video stream
process_video_stream(video_stream_url)

2. Sending Results to Mobile UI (Flask Web Server)

Now, we will create a simple Flask server to stream the results (bounding box positions, object statistics) to the mobile app. The mobile app can use this data to display on a user interface.

from flask import Flask, jsonify, Response
import cv2

app = Flask(__name__)

# Global variable to store the latest frame and object data
latest_frame = None
object_statistics = {}

# This endpoint will serve the video stream to the mobile app
def generate():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for streaming the video
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API to get object statistics
@app.route('/object_stats')
def get_object_stats():
    return jsonify(object_statistics)

def process_video_stream_for_flask(video_stream_url):
    cap = cv2.VideoCapture(video_stream_url)
    net, output_layers = load_yolo_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Extract object information
        boxes, confidences, class_ids = [], [], []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Set confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maxima suppression to remove duplicates
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Store object statistics
        object_statistics = {
            "detected_objects": len(indices),
            "object_positions": [boxes[i] for i in indices.flatten()]
        }

        # Draw bounding boxes
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        global latest_frame
        latest_frame = frame  # Update the latest frame

    cap.release()

if __name__ == '__main__':
    video_stream_url = 'http://192.168.10.1:8080/video'  # Replace with your drone stream URL
    # Start video processing in the background
    from threading import Thread
    Thread(target=process_video_stream_for_flask, args=(video_stream_url,)).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)

3. Mobile App (Flutter/React Native)

Your mobile application (built using Flutter or React Native) will need to:

    Connect to the Flask server to receive the live video feed (/video_feed).
    Fetch object statistics via an API (/object_stats) to display on the UI.

For example, in Flutter, you can use the http package to interact with the /object_stats endpoint and display the results.

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  Future<void> fetchObjectStats() async {
    final response = await http.get(Uri.parse('http://<flask_server_ip>:5000/object_stats'));
    if (response.statusCode == 200) {
      print('Object Stats: ${response.body}');
    } else {
      throw Exception('Failed to load object statistics');
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title

There was an error generating a response
