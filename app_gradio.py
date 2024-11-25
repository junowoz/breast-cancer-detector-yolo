import gradio as gr
from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO('model.pt')  # Replace with your model's path

def detect_image(image):
    """
    Perform object detection on an uploaded image.
    """
    results = model(image)
    result = results[0]
    annotated_frame = result.plot()  # Get the annotated frame
    return annotated_frame

def detect_camera():
    """
    Perform real-time object detection using the webcam.
    """
    cap = cv2.VideoCapture(0)  # 0 refers to the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        result = results[0]
        annotated_frame = result.plot()

        # Yield the annotated frame for Gradio's live feed
        yield annotated_frame

    cap.release()

# Create the Gradio interface
iface = gr.Blocks()

with iface:
    gr.Markdown("## Breast Cancer Detection with YOLO")
    gr.Markdown("Upload an ultrasound image or use the webcam for detection.")

    with gr.Tab("Image Detection"):
        image_input = gr.Image(type="numpy", label="Upload an Ultrasound Image")
        image_output = gr.Image(label="Detection Result")
        image_button = gr.Button("Detect")
        image_button.click(detect_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Real-Time Camera Detection"):
        camera_feed = gr.Video(label="Webcam Feed")
        camera_output = gr.Video(label="Detection Result")
        camera_button = gr.Button("Start Camera")
        camera_button.click(detect_camera, inputs=None, outputs=camera_output, live=True)

iface.launch()
