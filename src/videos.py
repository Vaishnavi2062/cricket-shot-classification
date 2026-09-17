import cv2
import os

folder = "cricketshot/train/pull"

for filename in os.listdir(folder):

    if filename.endswith(".avi"):

        video_path = os.path.join(folder, filename)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Could not open:", filename)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        print(
            filename,
            "|",
            width, "x", height,
            "| FPS:", fps,
            "| Frames:", frame_count,
            "| Duration:", round(duration, 2), "sec"
        )

        cap.release()