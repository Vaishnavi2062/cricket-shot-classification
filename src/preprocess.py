
import cv2
import numpy as np
import os


def load_video_frames(video_path, num_frames=24, save_frames=True):

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    # Get total number of frames
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError(
            f"No frames found in video: {video_path}"
        )

    # Select equally spaced frame positions
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frame_indices = set(frame_indices)

    frames = []
    current_frame = 0

    # Folder for saving frames
    if save_frames:

        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        output_folder = os.path.join(
            "frames",
            video_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

    # Read video sequentially
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Check if this frame is selected
        if current_frame in frame_indices:

            # Save original frame
            if save_frames:

                frame_filename = os.path.join(
                    output_folder,
                    f"frame_{len(frames) + 1:02d}.jpg"
                )

                cv2.imwrite(
                    frame_filename,
                    frame
                )

            # Convert BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize to 224 x 224
            frame = cv2.resize(
                frame,
                (224, 224)
            )

            # Convert pixel values to 0-1
            frame = frame.astype(
                np.float32
            ) / 255.0

            frames.append(frame)

        current_frame += 1

        # Stop after all selected frames have been read
        if len(frames) >= num_frames:
            break

    cap.release()

    # If fewer than 32 frames were obtained,
    # duplicate the last valid frame
    if len(frames) > 0 and len(frames) < num_frames:

        print(
            f"Warning: {video_path} "
            f"provided only {len(frames)} readable frames. "
            f"Duplicating the last frame."
        )

        last_frame = frames[-1]

        while len(frames) < num_frames:
            frames.append(
                last_frame.copy()
            )

    # If no frames could be read
    if len(frames) == 0:

        raise ValueError(
            f"No readable frames found in: {video_path}"
        )

    return np.array(
        frames,
        dtype=np.float32
    )

