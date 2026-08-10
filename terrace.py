import os

import cv2


CAMERA_NAMES = ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
VIDEO_FILES = [
    "terrace1-c0.avi",
    "terrace1-c1.avi",
    "terrace1-c2.avi",
    "terrace1-c3.avi",
]

VIDEO_FPS = 25
VIDEO_FRAME_COUNT = 5010

GRID_WIDTH = 30
GRID_HEIGHT = 44
WORLD_X_MIN = -0.5
WORLD_Y_MIN = -1.5
WORLD_X_MAX = 7.0
WORLD_Y_MAX = 9.5


def open_videos(dataset_path):
    video_folder = os.path.join(dataset_path, "videos")
    return [
        cv2.VideoCapture(os.path.join(video_folder, filename))
        for filename in VIDEO_FILES
    ]


def read_synchronized_frames(captures, frame_index):
    frames = []
    for capture in captures:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        _, frame = capture.read()
        frames.append(frame)
    return frames


def synchronized_frame_sequence(captures, frame_indices):
    previous_frame = None

    for frame_index in frame_indices:
        if previous_frame is None:
            for capture in captures:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            frames_to_skip = frame_index - previous_frame - 1
            for _ in range(frames_to_skip):
                for capture in captures:
                    capture.grab()

        frames = []
        for capture in captures:
            _, frame = capture.read()
            frames.append(frame)

        yield frame_index, frames
        previous_frame = frame_index


def position_id_to_world(position_id):
    column = position_id % GRID_WIDTH
    row = position_id // GRID_WIDTH

    cell_width = (WORLD_X_MAX - WORLD_X_MIN) / GRID_WIDTH
    cell_height = (WORLD_Y_MAX - WORLD_Y_MIN) / GRID_HEIGHT
    world_x = WORLD_X_MIN + (column + 0.5) * cell_width
    world_y = WORLD_Y_MIN + (row + 0.5) * cell_height
    return world_x, world_y


def load_ground_truth(dataset_path):
    annotation_path = os.path.join(
        dataset_path,
        "annotations",
        "gt_terrace1.txt",
    )
    with open(annotation_path, "r", encoding="utf-8") as annotation_file:
        lines = [line.strip() for line in annotation_file if line.strip()]

    header = [int(value) for value in lines[1].split()]
    frame_count, person_count, grid_width, grid_height = header[:4]
    step_size, first_frame, last_frame = header[4:7]

    frames = {}
    for frame_index, line in enumerate(lines[2:]):
        positions = [int(value) for value in line.split()]
        people = []

        for person_id, position_id in enumerate(positions):
            if position_id < 0:
                continue
            world_x, world_y = position_id_to_world(position_id)
            people.append(
                {
                    "person_id": person_id,
                    "position_id": position_id,
                    "world_x": world_x,
                    "world_y": world_y,
                }
            )

        if people:
            frames[frame_index] = people

    return {
        "frame_count": frame_count,
        "person_count": person_count,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "step_size": step_size,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frames": frames,
    }
