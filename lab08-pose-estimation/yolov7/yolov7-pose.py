import time
import torch
import cv2
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def pose_video(frame):
    img = letterbox(frame, input_size, stride=64, auto=True)[0]

    img = transforms.ToTensor()(img)
    img = torch.tensor(np.array([img.numpy()]))
    img = img.to(device)

    with torch.no_grad():
        t1 = time.time()
        output, _ = model(img)
        t2 = time.time()

        fps = 1 / (t2 - t1)

        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=1,
            nkpt=17,
            kpt_label=True
        )

        output = output_to_keypoint(output)

    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, fps


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

input_size = 256

videos = [
    "skydiving",
    "far-away"
]

save_name = videos[0]
file_name = save_name + ".mp4"
vid_path = "../media/" + file_name


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Selected Device:", device)


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

weights = torch.load(
    "yolov7-w6-pose.pt",
    map_location=device,
    weights_only=False
)

model = weights["model"]
model.float().eval()
model.to(device)


# ------------------------------------------------------------------
# Load video
# ------------------------------------------------------------------

cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    print("Cannot open video:", vid_path)
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()

if not ret:
    print("Cannot read video:", vid_path)
    cap.release()
    exit()

h, w, _ = frame.shape

out = cv2.VideoWriter(
    f"{save_name}_yolov7.avi",
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    fps,
    (w, h)
)

# return to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# ------------------------------------------------------------------
# Run pose estimation
# ------------------------------------------------------------------

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read frame. Exiting...")
            break

        img, fps_ = pose_video(frame)

        cv2.putText(
            img,
            "FPS: {:.2f}".format(fps_),
            (200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "YOLOv7",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Output", img)
        out.write(img)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()