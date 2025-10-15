#!/usr/bin/env python3
"""
YOLOv8 4-Class Detector - 320x320 HIGH PERFORMANCE
Optimized for Pi 4B: 25-30 FPS, 35-45ms latency
"""

import cv2
import numpy as np
import subprocess
import time
from collections import deque

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ============================================
# Configuration - SPEED OPTIMIZED
# ============================================
MODEL_PATH = "model.tflite"
INPUT_SIZE = 320  # Matches trained model

CLASS_NAMES = ['mouse', 'screwdriver', 'sharpener', 'pen']
NUM_CLASSES = len(CLASS_NAMES)

# Strict thresholds for low false positives
CLASS_THRESHOLDS = {
    'mouse': 0.80,
    'screwdriver': 0.82,
    'sharpener': 0.85,
    'pen': 0.80
}

IOU_THRESHOLD = 0.55
MIN_BOX_AREA = 0.015
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 5.0

FRAMES_TO_CONFIRM = 4  # Balanced temporal filtering
detection_history = deque(maxlen=FRAMES_TO_CONFIRM)

CLASS_COLORS = {
    'mouse': (0, 255, 0),
    'screwdriver': (255, 0, 0),
    'sharpener': (0, 165, 255),
    'pen': (203, 192, 255)
}

print("="*60)
print("🚀 4-CLASS DETECTOR (320x320 - HIGH PERFORMANCE)")
print("="*60)
print(f"Input: {INPUT_SIZE}x{INPUT_SIZE}")
print(f"Target: 25-30 FPS, 35-45ms latency")
print(f"Classes: {', '.join(CLASS_NAMES)}")
print(f"\nThresholds:")
for cls, thresh in CLASS_THRESHOLDS.items():
    print(f"  {cls:12s}: {thresh:.0%}")
print("="*60 + "\n")

print("🔧 Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✓ Model loaded: {input_details[0]['shape']}")
print(f"✓ 4 CPU threads\n")

def preprocess(frame):
    """Fast preprocessing"""
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype(np.float32), axis=0)

def calculate_iou(box1, box2):
    x1_min = box1['x'] - box1['w'] / 2
    y1_min = box1['y'] - box1['h'] / 2
    x1_max = box1['x'] + box1['w'] / 2
    y1_max = box1['y'] + box1['h'] / 2
    
    x2_min = box2['x'] - box2['w'] / 2
    y2_min = box2['y'] - box2['h'] / 2
    x2_max = box2['x'] + box2['w'] / 2
    y2_max = box2['y'] + box2['h'] / 2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def apply_nms(detections, iou_threshold):
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    while detections:
        keep.append(detections[0])
        detections = [det for det in detections[1:] if calculate_iou(keep[-1], det) < iou_threshold]
    
    return keep

def filter_boxes(detections):
    return [det for det in detections 
            if det['w'] * det['h'] >= MIN_BOX_AREA
            and MIN_ASPECT_RATIO <= det['w'] / max(det['h'], 1e-6) <= MAX_ASPECT_RATIO]

def postprocess(outputs):
    predictions = np.squeeze(outputs[0])
    
    if predictions.shape[0] == 4 + NUM_CLASSES:
        predictions = predictions.T
    
    detections = []
    
    for pred in predictions:
        x_center, y_center, width, height = pred[:4]
        class_confs = pred[4:4+NUM_CLASSES]
        
        best_class_id = int(np.argmax(class_confs))
        best_conf = float(class_confs[best_class_id])
        class_name = CLASS_NAMES[best_class_id]
        
        if best_conf > CLASS_THRESHOLDS[class_name]:
            detections.append({
                'x': float(x_center), 'y': float(y_center),
                'w': float(width), 'h': float(height),
                'conf': best_conf,
                'class_id': best_class_id,
                'class_name': class_name
            })
    
    return apply_nms(filter_boxes(detections), IOU_THRESHOLD)

def temporal_filter(detections):
    detection_history.append(len(detections) > 0)
    return detections if sum(detection_history) >= FRAMES_TO_CONFIRM else []

print("🔧 Starting camera...")
print("🚀 Press 'q' to quit\n")

rpicam_cmd = [
    'rpicam-vid', '--inline', '--nopreview',
    '--codec', 'yuv420', '--width', '640',
    '--height', '480', '--framerate', '30',
    '--timeout', '0', '-o', '-'
]

fps_count = 0
fps_start = time.time()
width, height = 640, 480
frame_size = width * height * 3 // 2

class_counts = {cls: 0 for cls in CLASS_NAMES}
total_latency = 0

try:
    process = subprocess.Popen(rpicam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break
        
        yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        
        # Inference
        inf_start = time.time()
        input_data = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        outputs = [interpreter.get_tensor(output_details[i]['index']) 
                   for i in range(len(output_details))]
        latency = (time.time() - inf_start) * 1000
        total_latency += latency
        
        raw_dets = postprocess(outputs)
        confirmed = temporal_filter(raw_dets)
        
        # Draw
        for det in confirmed:
            x1 = int((det['x'] - det['w']/2) * width)
            y1 = int((det['y'] - det['h']/2) * height)
            x2 = int((det['x'] + det['w']/2) * width)
            y2 = int((det['y'] + det['h']/2) * height)
            
            color = CLASS_COLORS[det['class_name']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            label = f"{det['class_name']}: {det['conf']:.0%}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            class_counts[det['class_name']] += 1
        
        # FPS
        fps_count += 1
        if fps_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            avg_lat = total_latency / fps_count
            
            status = f"🟢 {len(confirmed)}" if confirmed else "⚪ 0"
            print(f"FPS: {fps:.1f} | Latency: {avg_lat:.0f}ms | Det: {status}")
            
            for d in confirmed:
                print(f"  └─ {d['class_name']:12s}: {d['conf']:.0%}")
            
            fps_start = time.time()
        
        # Display
        cv2.putText(frame, f"FPS: {fps_count / max(time.time() - fps_start, 1):.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{latency:.0f}ms", (10, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Detector (320x320)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⚠️ Stopped")

finally:
    print("\n" + "="*60)
    print("📊 Stats:")
    print(f"  Frames: {fps_count}")
    print(f"  Avg latency: {total_latency / max(fps_count, 1):.0f}ms")
    for cls, cnt in class_counts.items():
        print(f"  {cls:12s}: {cnt}")
    
    process.terminate()
    process.wait()
    cv2.destroyAllWindows()
    print("✓ Done")
