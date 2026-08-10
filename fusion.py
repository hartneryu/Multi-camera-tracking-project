import math

import numpy as np


def distance_between(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def calculate_weighted_center(members):
    total_confidence = sum(member["confidence"] for member in members)

    world_x = sum(
        member["world_x"] * member["confidence"] for member in members
    ) / total_confidence
    world_y = sum(
        member["world_y"] * member["confidence"] for member in members
    ) / total_confidence
    return world_x, world_y


def calculate_weighted_embedding(members):
    members_with_embeddings = [
        member for member in members if member.get("embedding") is not None
    ]
    if not members_with_embeddings:
        return None

    embedding = sum(
        member["embedding"] * member["confidence"]
        for member in members_with_embeddings
    )
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def fuse_detections(
    detections,
    maximum_distance,
):
    sorted_detections = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )
    clusters = []

    for detection in sorted_detections:
        best_cluster = None
        best_distance = maximum_distance

        for cluster in clusters:
            camera_already_used = detection["camera_index"] in cluster["cameras"]
            if camera_already_used:
                continue

            distance = distance_between(
                detection["world_x"],
                detection["world_y"],
                cluster["world_x"],
                cluster["world_y"],
            )
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster

        if best_cluster is None:
            clusters.append(
                {
                    "world_x": detection["world_x"],
                    "world_y": detection["world_y"],
                    "confidence": detection["confidence"],
                    "cameras": {detection["camera_index"]},
                    "members": [detection],
                    "embedding": detection.get("embedding"),
                }
            )
            continue

        best_cluster["members"].append(detection)
        best_cluster["cameras"].add(detection["camera_index"])
        center_x, center_y = calculate_weighted_center(best_cluster["members"])
        best_cluster["world_x"] = center_x
        best_cluster["world_y"] = center_y
        best_cluster["confidence"] = sum(
            member["confidence"] for member in best_cluster["members"]
        ) / len(best_cluster["members"])
        best_cluster["embedding"] = calculate_weighted_embedding(
            best_cluster["members"]
        )

    return clusters
