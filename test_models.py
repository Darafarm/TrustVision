import cv2
from ultralytics import YOLO
import time

def test_model(model_name):
    model = YOLO(model_name)
    cap = cv2.VideoCapture('test_video.mp4')
    frame_count = 0
    fps_list = []

    print(f"\nTesting {model_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        results = model(frame, verbose=False)
        latency = (time.time() - start) * 1000
        fps = 1000 / latency if latency > 0 else 0
        fps_list.append(fps)
        frame_count += 1

    cap.release()
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_lat = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Frames: {frame_count} | Avg FPS: {avg_fps:.1f} | Avg Latency: {1000/avg_fps:.1f}ms")

test_model('yolo11n.pt')
test_model('yolo11s.pt')
test_model('yolov8n.pt')

print("\nModel comparison complete!")