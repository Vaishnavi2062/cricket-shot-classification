import cv2
import os
import statistics

dataset_path = "cricketshot"

frame_counts = []
durations = []

for split in ["train", "val", "test"]:

    split_path = os.path.join(dataset_path, split)

    for class_name in os.listdir(split_path):

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):

            if not filename.endswith(".avi"):
                continue

            video_path = os.path.join(class_path, filename)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Could not open:", video_path)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            duration = frame_count / fps if fps > 0 else 0

            frame_counts.append(frame_count)
            durations.append(duration)

            cap.release()


print("\n     DATASET VIDEO STATISTICS     ")

print("Total videos:", len(frame_counts))

print("Minimum frames:", min(frame_counts))
print("Maximum frames:", max(frame_counts))
print("Average frames:", round(statistics.mean(frame_counts), 2))
print("Median frames:", statistics.median(frame_counts))

print("\nMinimum duration:", round(min(durations), 2), "seconds")
print("Maximum duration:", round(max(durations), 2), "seconds")
print("Average duration:", round(statistics.mean(durations), 2), "seconds")