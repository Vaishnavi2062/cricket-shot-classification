import cv2
import os

video_path = "dataset/cricketshot/train/cover/cover_0001.avi"

output_folder = "nth_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

nth = 5

print("Total frames:", total_frames)
print("Taking every", nth, "th frame")

frame_number = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % nth == 0:

        frame_path = os.path.join(
            output_folder,
            f"frame_{saved_count:03d}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        saved_count += 1

    frame_number += 1

cap.release()

print("Done!")
print("Frames extracted:", saved_count)