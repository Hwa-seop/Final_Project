from flask import Flask, render_template, Response, jsonify, request
import cv2


pipeline = (
    'v4l2src device=/dev/video0 ! videoconvert ! '
    'video/x-raw,width=640,height=480,framerate=30/1 ! appsink'
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("🚫 GStreamer 파이프라인 열기 실패")
else:
    print("✅ GStreamer 파이프라인 정상 작동")

cap.release()


hwaseop@hwaseop-virtual-machine:~/final$ v4l2-ctl --list-devices
Web Camera: Web Camera (usb-0000:03:00.0-2):
        /dev/video0
        /dev/video1
        /dev/media0

hwaseop@hwaseop-virtual-machine:~/final$ python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
    GStreamer:                   NO