import cv2
from ultralytics import YOLO
import time
import psutil

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture('test_video.mp4')

frame_count = 0
fps_list = []

print("Starting detection test (no tracking)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame, verbose=False)
    latency = (time.time() - start) * 1000
    fps = 1000 / latency if latency > 0 else 0
    fps_list.append(fps)

    annotated = results[0].plot()
    cv2.imshow('TrustVision - Detection Only', annotated)

    if frame_count % 30 == 0:
        cpu = psutil.cpu_percent()
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        print(f"Frame {frame_count} | FPS: {avg_fps:.1f} | Latency: {latency:.0f}ms | CPU: {cpu:.0f}%")

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
print(f"\n--- Detection Summary ---")
print(f"Total frames processed: {frame_count}")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Average latency: {sum(fps_list)/len(fps_list):.1f} ms")