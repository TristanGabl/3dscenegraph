import cv2
import os


def create_video_from_frames(frame_paths, output_dir, output_video_path, fps=10):
    if not frame_paths:
        raise ValueError("No frames provided to create the video.")

    # Read the first frame to get the dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        video.write(frame)

    video.release()
    print(f"Video saved at {output_video_path}")

input = "output/scene0134_02"
type = "fused_votes"

output_dir = os.path.join(input, "video")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_paths = []
frame_path = [os.path.join(input, "full", file) for file in os.listdir(os.path.join(input, "full")) if file.endswith('.jpg') and type in file]
frame_paths.extend(frame_path)
frame_paths.sort()  # Sort the frames to maintain the order

output_video_path = os.path.join(output_dir, f"{type}.mp4")

create_video_from_frames(frame_paths, output_dir, output_video_path, fps=10)

