import cv2
import streamlit as st
import numpy as np

import os
import subprocess

def detect_traffic_signals(frame, yolo_weights, yolo_config, yolo_classes):
    # Load YOLO model
    net = cv2.dnn.readNet(yolo_weights, yolo_config)

    # Load COCO names file
    with open(yolo_classes, 'r') as f:
        classes = f.read().strip().split('\n')

    height, width = frame.shape[:2]

    # Convert frame to blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass
    detections = net.forward(output_layer_names)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'traffic light':
                # Draw bounding box around traffic signal
                box = obj[0:4] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype("int")
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    return frame



# Function to process the video and apply traffic signal detection
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    # Get video properties
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a VideoWriter object to save the processed video
    output_path = "output.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    st.text("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect traffic signals in the frame
        processed_frame = detect_traffic_signals(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

    st.text("Video processing complete.")

    # Release video capture and writer objects
    cap.release()
    out.release()

    return output_path

# Streamlit app
def main():
    st.title("Traffic Signal Detection App")

    # Upload video through Streamlit
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    if uploaded_file:
        st.video(uploaded_file)
        save_path = os.path.join("saved_videos", uploaded_file.name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as video_file:
            video_file.write(uploaded_file.read())
        st.markdown(f"[Download the saved video]({save_path})")
        os.system(f"python ./ultralytics/ultralytics/yolo/v8/detect/predict.py model='best.pt' source='{save_path}'")
        



if __name__ == "__main__":
    main()
