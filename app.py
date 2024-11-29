import gradio as gr
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('model.pt')  # Replace with the path to your model

def detect(image):
    results = model(image)
    result = results[0]
    annotated_frame = result.plot()
    return annotated_frame

iface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs="image",
    title="Detecção de Câncer de Mama com YOLO",
    description="Faça upload de uma imagem de ultrassom para detectar tumores."
)

iface.launch()
