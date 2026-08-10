import cv2
import numpy as np

from terrace import (
    CAMERA_NAMES,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
)
from projection import world_to_image


PANEL_WIDTH = 640
PANEL_HEIGHT = 512


def person_color(person_id):
    blue = 80 + (person_id * 67) % 176
    green = 80 + (person_id * 37) % 176
    red = 80 + (person_id * 97) % 176
    return blue, green, red


def draw_title(panel, text):
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 42), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def camera_panel(image, camera_index, frame_index):
    panel = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    title = CAMERA_NAMES[camera_index] + " | frame " + str(frame_index)
    draw_title(panel, title)
    return panel


def projection_camera_panel(
    image,
    camera_index,
    people,
    calibration,
    frame_index,
):
    image_height, image_width = image.shape[:2]
    panel = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    scale_x = PANEL_WIDTH / image_width
    scale_y = PANEL_HEIGHT / image_height
    projected_count = 0

    for person in people:
        foot = world_to_image(
            person["world_x"],
            person["world_y"],
            0.0,
            calibration,
            image_width,
            image_height,
        )
        head = world_to_image(
            person["world_x"],
            person["world_y"],
            1.75,
            calibration,
            image_width,
            image_height,
        )
        if foot is None or head is None:
            continue
        if not (0 <= foot[0] < image_width and 0 <= foot[1] < image_height):
            continue

        projected_count += 1
        foot_point = (int(foot[0] * scale_x), int(foot[1] * scale_y))
        head_point = (int(head[0] * scale_x), int(head[1] * scale_y))
        color = person_color(person["person_id"])
        cv2.line(panel, head_point, foot_point, color, 3, cv2.LINE_AA)
        cv2.circle(panel, foot_point, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, foot_point, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            "ID " + str(person["person_id"]),
            (head_point[0] + 6, max(55, head_point[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    title = (
        CAMERA_NAMES[camera_index]
        + " | "
        + str(projected_count)
        + " projected positions"
    )
    draw_title(panel, title)
    return panel


def world_to_panel(world_x, world_y, margin=60):
    usable_width = PANEL_WIDTH - 2 * margin
    usable_height = PANEL_HEIGHT - 2 * margin
    x_fraction = (world_x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN)
    y_fraction = (world_y - WORLD_Y_MIN) / (WORLD_Y_MAX - WORLD_Y_MIN)
    panel_x = int(margin + x_fraction * usable_width)
    panel_y = int(PANEL_HEIGHT - margin - y_fraction * usable_height)
    return panel_x, panel_y


def bird_eye_panel(people, frame_index):
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 245, dtype=np.uint8)
    margin = 60
    cv2.rectangle(
        panel,
        (margin, margin),
        (PANEL_WIDTH - margin, PANEL_HEIGHT - margin),
        (255, 255, 255),
        -1,
    )

    for world_x in range(0, 8):
        start = world_to_panel(world_x, WORLD_Y_MIN, margin)
        end = world_to_panel(world_x, WORLD_Y_MAX, margin)
        cv2.line(panel, start, end, (220, 220, 220), 1)

    for world_y in range(-1, 10):
        start = world_to_panel(WORLD_X_MIN, world_y, margin)
        end = world_to_panel(WORLD_X_MAX, world_y, margin)
        cv2.line(panel, start, end, (220, 220, 220), 1)

    cv2.rectangle(
        panel,
        (margin, margin),
        (PANEL_WIDTH - margin, PANEL_HEIGHT - margin),
        (40, 40, 40),
        2,
    )

    for person in people:
        point = world_to_panel(person["world_x"], person["world_y"], margin)
        color = person_color(person["person_id"])
        cv2.circle(panel, point, 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, point, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            "ID " + str(person["person_id"]),
            (point[0] + 9, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    draw_title(
        panel,
        "Ground-truth BEV | " + str(len(people)) + " people | frame " + str(frame_index),
    )
    return panel


def create_terrace_view(frames, people, frame_index):
    panels = [
        camera_panel(frame, camera_index, frame_index)
        for camera_index, frame in enumerate(frames)
    ]
    bev = bird_eye_panel(people, frame_index)
    blank = np.full_like(bev, 245)
    first_row = cv2.hconcat([panels[0], panels[1], bev])
    second_row = cv2.hconcat([panels[2], panels[3], blank])
    return cv2.vconcat([first_row, second_row])


def create_projection_validation_view(
    frames,
    people,
    calibrations,
    frame_index,
):
    panels = [
        projection_camera_panel(
            frame,
            camera_index,
            people,
            calibrations[camera_index],
            frame_index,
        )
        for camera_index, frame in enumerate(frames)
    ]
    bev = bird_eye_panel(people, frame_index)
    blank = np.full_like(bev, 245)
    cv2.putText(
        blank,
        "Lines: projected 1.75 m person height",
        (80, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blank,
        "Dots: projected ground contact points",
        (80, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    first_row = cv2.hconcat([panels[0], panels[1], bev])
    second_row = cv2.hconcat([panels[2], panels[3], blank])
    return cv2.vconcat([first_row, second_row])


def yolo_camera_panel(image, camera_index, detections):
    image_height, image_width = image.shape[:2]
    panel = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    scale_x = PANEL_WIDTH / image_width
    scale_y = PANEL_HEIGHT / image_height

    for detection in detections:
        x1 = int(detection["x1"] * scale_x)
        y1 = int(detection["y1"] * scale_y)
        x2 = int(detection["x2"] * scale_x)
        y2 = int(detection["y2"] * scale_y)
        foot = ((x1 + x2) // 2, y2)
        cv2.rectangle(
            panel,
            (x1, y1),
            (x2, y2),
            (0, 130, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(panel, foot, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, foot, 3, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            format(detection["confidence"], ".2f"),
            (x1, max(55, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 130, 255),
            2,
            cv2.LINE_AA,
        )

    draw_title(
        panel,
        CAMERA_NAMES[camera_index]
        + " | "
        + str(len(detections))
        + " YOLO detections",
    )
    return panel


def yolo_bird_eye_panel(
    projected_detections,
    fused_detections,
    people,
    frame_index,
):
    panel = bird_eye_panel([], frame_index)
    camera_colors = [
        (30, 80, 235),
        (40, 180, 40),
        (230, 110, 30),
        (190, 50, 190),
    ]

    for person in people:
        point = world_to_panel(person["world_x"], person["world_y"])
        cv2.drawMarker(
            panel,
            point,
            (30, 160, 30),
            cv2.MARKER_CROSS,
            16,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "GT " + str(person["person_id"]),
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 130, 20),
            1,
            cv2.LINE_AA,
        )

    for detection in projected_detections:
        point = world_to_panel(detection["world_x"], detection["world_y"])
        color = camera_colors[detection["camera_index"]]
        cv2.circle(panel, point, 4, color, -1, cv2.LINE_AA)

    for observation in fused_detections:
        point = world_to_panel(observation["world_x"], observation["world_y"])
        cv2.circle(panel, point, 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            str(len(observation["cameras"])),
            (point[0] + 8, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    draw_title(
        panel,
        "YOLO BEV | "
        + str(len(projected_detections))
        + " camera points -> "
        + str(len(fused_detections))
        + " fused",
    )
    cv2.putText(
        panel,
        "cross: ground truth | colored: camera | white: fused",
        (85, 495),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return panel


def create_yolo_view(
    frames,
    camera_detections,
    projected_detections,
    fused_detections,
    people,
    frame_index,
):
    panels = [
        yolo_camera_panel(frame, camera_index, camera_detections[camera_index])
        for camera_index, frame in enumerate(frames)
    ]
    bev = yolo_bird_eye_panel(
        projected_detections,
        fused_detections,
        people,
        frame_index,
    )
    blank = np.full_like(bev, 245)
    cv2.putText(
        blank,
        "Bounding-box bottom center -> ground plane",
        (70, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blank,
        "Nearby points from different cameras -> one observation",
        (35, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    first_row = cv2.hconcat([panels[0], panels[1], bev])
    second_row = cv2.hconcat([panels[2], panels[3], blank])
    return cv2.vconcat([first_row, second_row])


def tracking_camera_panel(image, camera_index, detections):
    image_height, image_width = image.shape[:2]
    panel = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    scale_x = PANEL_WIDTH / image_width
    scale_y = PANEL_HEIGHT / image_height

    for detection in detections:
        x1 = int(detection["x1"] * scale_x)
        y1 = int(detection["y1"] * scale_y)
        x2 = int(detection["x2"] * scale_x)
        y2 = int(detection["y2"] * scale_y)
        track_id = detection.get("track_id")
        confirmed = detection.get("track_confirmed", False)
        color = person_color(track_id) if confirmed else (140, 140, 140)
        thickness = 3 if confirmed else 1
        cv2.rectangle(
            panel,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )
        if confirmed:
            cv2.putText(
                panel,
                "ID " + str(track_id),
                (x1, max(55, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    confirmed_count = len(
        [
            detection
            for detection in detections
            if detection.get("track_confirmed", False)
        ]
    )
    draw_title(
        panel,
        CAMERA_NAMES[camera_index]
        + " | "
        + str(confirmed_count)
        + " tracked boxes",
    )
    return panel


def draw_covariance_ellipse(panel, track, color, margin=60):
    world_width = WORLD_X_MAX - WORLD_X_MIN
    world_height = WORLD_Y_MAX - WORLD_Y_MIN
    scale = np.array(
        [
            [(PANEL_WIDTH - 2 * margin) / world_width, 0.0],
            [0.0, -(PANEL_HEIGHT - 2 * margin) / world_height],
        ]
    )
    pixel_covariance = scale @ track["covariance"][:2, :2] @ scale.T
    values, vectors = np.linalg.eigh(pixel_covariance)
    values = np.maximum(values, 0.0)
    largest_first = np.argsort(values)[::-1]
    values = values[largest_first]
    vectors = vectors[:, largest_first]

    # 5.991 is the 95% chi-square boundary for a two-dimensional position.
    axes = np.sqrt(5.991 * values)
    axes = (max(1, int(axes[0])), max(1, int(axes[1])))
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    center = world_to_panel(track["world_x"], track["world_y"], margin)
    cv2.ellipse(
        panel,
        center,
        axes,
        angle,
        0,
        360,
        color,
        1,
        cv2.LINE_AA,
    )


def tracking_bird_eye_panel(tracks, people, frame_index):
    panel = bird_eye_panel([], frame_index)

    for person in people:
        point = world_to_panel(person["world_x"], person["world_y"])
        cv2.drawMarker(
            panel,
            point,
            (30, 160, 30),
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )

    confirmed_tracks = [track for track in tracks if track["confirmed"]]
    for track in confirmed_tracks:
        color = person_color(track["id"])
        draw_covariance_ellipse(panel, track, color)
        history = [
            world_to_panel(world_x, world_y)
            for world_x, world_y in track["history"]
        ]
        if len(history) >= 2:
            cv2.polylines(
                panel,
                [np.array(history, dtype=np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )

        point = world_to_panel(track["world_x"], track["world_y"])
        point_color = color if track["visible"] else (140, 140, 140)
        cv2.circle(panel, point, 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(panel, point, 6, point_color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            "ID " + str(track["id"]),
            (point[0] + 8, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            point_color,
            2,
            cv2.LINE_AA,
        )

    for track in tracks:
        if track["confirmed"]:
            continue
        point = world_to_panel(track["world_x"], track["world_y"])
        cv2.circle(panel, point, 3, (160, 160, 160), -1, cv2.LINE_AA)

    draw_title(
        panel,
        "Kalman BEV | "
        + str(len(confirmed_tracks))
        + " confirmed tracks | frame "
        + str(frame_index),
    )
    cv2.putText(
        panel,
        "ellipse: 95% uncertainty | gray: predicted or tentative",
        (90, 495),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return panel


def create_tracking_view(
    frames,
    camera_detections,
    tracks,
    people,
    frame_index,
):
    panels = [
        tracking_camera_panel(
            frame,
            camera_index,
            camera_detections[camera_index],
        )
        for camera_index, frame in enumerate(frames)
    ]
    bev = tracking_bird_eye_panel(tracks, people, frame_index)
    blank = np.full_like(bev, 245)
    cv2.putText(
        blank,
        "Kalman prediction + Mahalanobis/Hungarian matching",
        (45, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blank,
        "Multi-camera initialization + single-camera continuation",
        (50, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    first_row = cv2.hconcat([panels[0], panels[1], bev])
    second_row = cv2.hconcat([panels[2], panels[3], blank])
    return cv2.vconcat([first_row, second_row])
