import os
import sys
import argparse
import glob
import time
import platform

import cv2
import numpy as np
from super_gradients.training import models
from super_gradients.common.object_names import Models

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', help='Image source: image file, folder, video file, or usb index (e.g., usb0)', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold (e.g., 0.4)', default=0.5)
parser.add_argument('--resolution', help='Resolution WxH (e.g., 640x480)', default=None)
parser.add_argument('--record', help='Record video output to demo1.avi (only for video/usb input)', action='store_true')
args = parser.parse_args()

img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Load YOLO-NAS model
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
labels = model._class_names

img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print(f'Invalid source input: {img_source}')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.lower().split('x'))

if record:
    if source_type not in ['video', 'usb'] or not user_res:
        print('Recording requires video/usb source and --resolution')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Done processing images.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            print('End of stream or failed to grab frame.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    prediction = model.predict(frame, conf=min_thresh)[0].prediction

    object_count = 0
    for pred in prediction:
        xmin, ymin, xmax, ymax, conf, class_id = map(int, pred[:4]) + [pred[4], int(pred[5])]
        if conf > min_thresh:
            color = bbox_colors[class_id % len(bbox_colors)]
            label = f"{labels[class_id]}: {int(conf * 100)}%"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            object_count += 1

    if source_type in ['video', 'usb']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.imshow('YOLO-NAS Results', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type not in ['image', 'folder'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
print(f'Average FPS: {avg_frame_rate:.2f}')
