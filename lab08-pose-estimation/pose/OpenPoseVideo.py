import cv2
import time
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="far-away-fast.mp4", help="Input Video")
args = parser.parse_args()

MODE = "MPI"

protoFile = "./mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./mpi/pose_iter_160000.caffemodel"

nPoints = 15
POSE_PAIRS = [
    [0,1], [1,2], [2,3], [3,4],
    [1,5], [5,6], [6,7],
    [1,14], [14,8], [8,9], [9,10],
    [14,11], [11,12], [12,13]
]

inWidth = 96
inHeight = 96
threshold = 0.1
MAX_FRAMES = 5

current_folder = os.path.dirname(os.path.abspath(__file__))
input_source = os.path.join(current_folder, "..", "media", args.video_file)

print("Looking for video:", input_source)

if not os.path.exists(input_source):
    print("ERROR: Video not found.")
    sys.exit()

cap = cv2.VideoCapture(input_source)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    sys.exit()

hasFrame, frame = cap.read()

if not hasFrame:
    print("ERROR: Cannot read first frame.")
    sys.exit()

save_name = os.path.splitext(os.path.basename(input_source))[0]
print("Saving as:", save_name + "_openpose.avi")

vid_writer = cv2.VideoWriter(
    f"{save_name}_openpose.avi",
    cv2.VideoWriter_fourcc('M','J','P','G'),
    10,
    (frame.shape[1], frame.shape[0])
)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")

frame_count = 0

while True:
    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Finished video.")
        break

    frame_count += 1

    if frame_count > MAX_FRAMES:
        print("Stopped after 5 frames for quick submission.")
        break

    t = time.time()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(
        frame,
        1.0 / 255,
        (inWidth, inHeight),
        (0, 0, 0),
        swapRB=False,
        crop=False
    )

    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)
            cv2.circle(frame, points[partA], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[partB], 5, (0, 0, 255), -1)

    cv2.putText(
        frame,
        "OpenPose CPU | time = {:.2f}s".format(time.time() - t),
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 50, 0),
        2
    )
    cv2.imshow("OpenPose Output", frame)
    cv2.waitKey(7)

    vid_writer.write(frame)

    print(f"Processed frame {frame_count}")

cap.release()
vid_writer.release()

print("Done. Output saved:", save_name + "_openpose.avi")
