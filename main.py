import argparse  # noqa: I001
import os

import cv2
import numpy as np

from appearance import extract_appearance_embeddings, load_appearance_model
from detector import detect_people_batch, load_detector
from evaluation import (
    create_evaluator,
    evaluate_frame,
    save_evaluation,
    summarize_evaluator,
)
from fusion import fuse_detections
from projection import (
    calculate_round_trip_errors,
    load_camera_calibrations,
    project_box_foot,
)
from terrace import (
    VIDEO_FPS,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
    load_ground_truth,
    open_videos,
    read_synchronized_frames,
    synchronized_frame_sequence,
)
from tracker import (
    APPEARANCE_WEIGHT,
    MAXIMUM_MISSED_SECONDS,
    create_tracker,
    update_tracker,
)
from visualization import (
    create_projection_validation_view,
    create_tracking_view,
    create_terrace_view,
    create_yolo_view,
)


DETECTOR_MODEL = "yolo11s.pt"
DETECTION_CONFIDENCE = 0.25
DETECTION_IMAGE_SIZE = 640
FUSION_DISTANCE = 1.2
MINIMUM_CAMERAS = 2
PROCESSING_FPS = 5.0
MATCHING_DISTANCE = 0.5
WARMUP_SECONDS = 1.0
APPEARANCE_MODEL = "models/osnet_x0_25_msmt17.pth"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display synchronized EPFL Terrace cameras and ground truth"
    )
    parser.add_argument("--dataset", default="data/EPFL-Terrace")
    parser.add_argument("--frame", type=int, default=2000)
    parser.add_argument("--output")
    parser.add_argument(
        "--mode",
        choices=["view", "projection", "yolo", "track", "evaluate"],
        default="projection",
    )
    parser.add_argument("--start-frame", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=5.0)
    return parser.parse_args()


def run_yolo(
    frames,
    calibrations,
    detector=None,
    appearance_model=None,
):
    if detector is None:
        detector = load_detector(DETECTOR_MODEL)
    camera_detections = detect_people_batch(
        detector,
        frames,
        DETECTION_CONFIDENCE,
        DETECTION_IMAGE_SIZE,
    )

    if appearance_model is not None:
        for frame, detections in zip(frames, camera_detections):
            embeddings = extract_appearance_embeddings(
                appearance_model,
                frame,
                detections,
            )
            for detection, embedding in zip(detections, embeddings):
                detection["embedding"] = embedding

    projected_detections = []

    for camera_index, (frame, detections) in enumerate(
        zip(frames, camera_detections)
    ):
        image_height, image_width = frame.shape[:2]

        for detection in detections:
            box = [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            ]
            world_point = project_box_foot(
                box,
                calibrations[camera_index],
                image_width,
                image_height,
            )
            if world_point is None:
                continue
            world_x, world_y = world_point
            if not (
                WORLD_X_MIN <= world_x <= WORLD_X_MAX
                and WORLD_Y_MIN <= world_y <= WORLD_Y_MAX
            ):
                continue

            projected_detection = detection.copy()
            projected_detection["camera_index"] = camera_index
            projected_detection["world_x"] = world_x
            projected_detection["world_y"] = world_y
            projected_detection["source_detection"] = detection
            projected_detections.append(projected_detection)

    fused_detections = fuse_detections(
        projected_detections,
        FUSION_DISTANCE,
    )
    multi_camera_observations = [
        observation
        for observation in fused_detections
        if len(observation["cameras"]) >= MINIMUM_CAMERAS
    ]
    single_camera_observations = [
        observation
        for observation in fused_detections
        if len(observation["cameras"]) == 1
    ]
    return (
        camera_detections,
        projected_detections,
        multi_camera_observations,
        single_camera_observations,
    )


def tracking_frame_indices(start_frame, seconds):
    sample_count = max(1, int(seconds * PROCESSING_FPS))
    indices = [
        round(start_frame + sample_index * VIDEO_FPS / PROCESSING_FPS)
        for sample_index in range(sample_count)
    ]
    return sorted(set(indices))


def run_tracking_demo(args):
    if args.output is None:
        args.output = "outputs/terrace_tracking.mp4"
    output_folder = os.path.dirname(args.output)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    indices = tracking_frame_indices(args.start_frame, args.seconds)
    calibrations = load_camera_calibrations(args.dataset)
    ground_truth = load_ground_truth(args.dataset)["frames"]
    detector = load_detector(DETECTOR_MODEL)
    appearance_model = load_appearance_model(APPEARANCE_MODEL)
    tracker = create_tracker()
    captures = open_videos(args.dataset)
    writer = None
    previous_frame = None

    try:
        sequence = synchronized_frame_sequence(captures, indices)
        for sample_index, (frame_index, frames) in enumerate(sequence):
            (
                camera_detections,
                _,
                observations,
                single_camera_observations,
            ) = run_yolo(
                frames,
                calibrations,
                detector,
                appearance_model,
            )
            if previous_frame is None:
                dt = 1.0 / PROCESSING_FPS
            else:
                dt = (frame_index - previous_frame) / VIDEO_FPS
            previous_frame = frame_index

            tracks = update_tracker(
                tracker,
                observations,
                dt,
                weak_observations=single_camera_observations,
            )
            result = create_tracking_view(
                frames,
                camera_detections,
                tracks,
                ground_truth.get(frame_index, []),
                frame_index,
            )

            if writer is None:
                height, width = result.shape[:2]
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    PROCESSING_FPS,
                    (width, height),
                )
            writer.write(result)

            if sample_index % 10 == 0 or sample_index == len(indices) - 1:
                confirmed = len(
                    [track for track in tracks if track["confirmed"]]
                )
                print(
                    "Processed",
                    str(sample_index + 1) + "/" + str(len(indices)),
                    "| frame",
                    frame_index,
                    "| observations",
                    len(observations),
                    "| confirmed tracks",
                    confirmed,
                )
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()

    print("Saved:", args.output)
    print("Track candidates created:", tracker["created_ids"])
    print("Confirmed global IDs:", len(tracker["confirmed_ids"]))


def run_evaluation(args):
    if args.output is None:
        args.output = "outputs/terrace_evaluation.csv"
    output_folder = os.path.dirname(args.output)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    ground_truth = load_ground_truth(args.dataset)
    evaluation_indices = tracking_frame_indices(args.start_frame, args.seconds)
    first_evaluation_frame = evaluation_indices[0]
    last_evaluation_frame = evaluation_indices[-1]

    warmup_start = max(
        0,
        round(first_evaluation_frame - WARMUP_SECONDS * VIDEO_FPS),
    )
    warmup_duration = (first_evaluation_frame - warmup_start) / VIDEO_FPS
    warmup_indices = tracking_frame_indices(warmup_start, warmup_duration)
    annotation_frames = range(
        ground_truth["first_frame"],
        ground_truth["last_frame"] + 1,
        ground_truth["step_size"],
    )
    indices = sorted(
        set(evaluation_indices)
        | set(warmup_indices)
        | {
            frame_index
            for frame_index in annotation_frames
            if first_evaluation_frame <= frame_index <= last_evaluation_frame
        }
    )
    calibrations = load_camera_calibrations(args.dataset)
    detector = load_detector(DETECTOR_MODEL)
    appearance_model = load_appearance_model(APPEARANCE_MODEL)
    tracker = create_tracker()
    evaluator = create_evaluator()
    captures = open_videos(args.dataset)
    previous_frame = None

    try:
        sequence = synchronized_frame_sequence(captures, indices)
        for sample_index, (frame_index, frames) in enumerate(sequence):
            _, _, observations, single_camera_observations = run_yolo(
                frames,
                calibrations,
                detector,
                appearance_model,
            )
            if previous_frame is None:
                dt = 1.0 / PROCESSING_FPS
            else:
                dt = (frame_index - previous_frame) / VIDEO_FPS
            previous_frame = frame_index
            tracks = update_tracker(
                tracker,
                observations,
                dt,
                weak_observations=single_camera_observations,
            )

            is_annotated = (
                frame_index >= first_evaluation_frame
                and frame_index <= last_evaluation_frame
                and (frame_index - ground_truth["first_frame"])
                % ground_truth["step_size"]
                == 0
            )
            if is_annotated:
                evaluate_frame(
                    evaluator,
                    frame_index,
                    ground_truth["frames"].get(frame_index, []),
                    tracks,
                    MATCHING_DISTANCE,
                )

            if sample_index % 25 == 0 or sample_index == len(indices) - 1:
                print(
                    "Processed",
                    str(sample_index + 1) + "/" + str(len(indices)),
                    "| frame",
                    frame_index,
                    "| evaluated labels",
                    len(evaluator["frames"]),
                )
    finally:
        for capture in captures:
            capture.release()

    summary = summarize_evaluator(evaluator)
    summary["processing_fps"] = PROCESSING_FPS
    summary["start_frame"] = evaluator["frames"][0]["frame"]
    summary["end_frame"] = evaluator["frames"][-1]["frame"]
    summary["track_candidates"] = tracker["created_ids"]
    summary["confirmed_global_ids"] = len(tracker["confirmed_ids"])
    summary["appearance"] = "osnet"
    summary["single_camera_continuation"] = True
    summary["appearance_weight"] = APPEARANCE_WEIGHT
    summary["maximum_missed_seconds"] = MAXIMUM_MISSED_SECONDS
    frame_path = save_evaluation(evaluator, summary, args.output)
    print("\nEvaluation summary")
    for name, value in summary.items():
        if isinstance(value, float):
            print(name + ":", format(value, ".4f"))
        else:
            print(name + ":", value)
    print("Saved:", args.output)
    print("Per-frame results:", frame_path)


def main():
    args = parse_arguments()
    if args.mode == "track":
        run_tracking_demo(args)
        return
    if args.mode == "evaluate":
        run_evaluation(args)
        return

    captures = open_videos(args.dataset)

    try:
        frames = read_synchronized_frames(captures, args.frame)
    finally:
        for capture in captures:
            capture.release()

    ground_truth = load_ground_truth(args.dataset)
    people = ground_truth["frames"].get(args.frame, [])
    if args.output is None:
        args.output = (
            "outputs/terrace_"
            + args.mode
            + "_"
            + str(args.frame)
            + ".png"
        )

    if args.mode == "projection":
        calibrations = load_camera_calibrations(args.dataset)
        result = create_projection_validation_view(
            frames,
            people,
            calibrations,
            args.frame,
        )
        image_height, image_width = frames[0].shape[:2]
        errors = calculate_round_trip_errors(
            people,
            calibrations,
            image_width,
            image_height,
        )
        if errors:
            print("Projection round-trip points:", len(errors))
            print("Mean round-trip error (m):", format(np.mean(errors), ".10f"))
            print("Maximum round-trip error (m):", format(max(errors), ".10f"))
    elif args.mode == "yolo":
        calibrations = load_camera_calibrations(args.dataset)
        (
            camera_detections,
            projected_detections,
            fused_detections,
            single_camera_observations,
        ) = run_yolo(frames, calibrations)
        result = create_yolo_view(
            frames,
            camera_detections,
            projected_detections,
            fused_detections,
            people,
            args.frame,
        )
        print(
            "Camera detections:",
            [len(detections) for detections in camera_detections],
        )
        print("Projected detections inside BEV:", len(projected_detections))
        print("Fused observations:", len(fused_detections))
        print("Single-camera clusters:", len(single_camera_observations))
    else:
        result = create_terrace_view(frames, people, args.frame)

    output_folder = os.path.dirname(args.output)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cv2.imwrite(args.output, result)

    print("Saved:", args.output)
    print("Frame:", args.frame)
    print("Ground-truth people:", len(people))


if __name__ == "__main__":
    main()
