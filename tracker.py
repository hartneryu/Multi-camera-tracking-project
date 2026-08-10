import numpy as np
from scipy.optimize import linear_sum_assignment


MEASUREMENT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)

REACTIVATION_APPEARANCE_WEIGHT = 0.8
REACTIVATION_APPEARANCE_DISTANCE = 0.3
APPEARANCE_WEIGHT = 0.5
MAXIMUM_APPEARANCE_DISTANCE = 0.5
MAXIMUM_MISSED_SECONDS = 5.0
MAHALANOBIS_GATE = 9.21
MINIMUM_HITS = 3
HISTORY_LENGTH = 50
APPEARANCE_MOMENTUM = 0.9


def create_tracker():
    return {
        "tracks": [],
        "next_id": 1,
        "created_ids": 0,
        "confirmed_ids": set(),
    }


def camera_count(observation):
    return max(1, len(observation.get("cameras", [])))


def measurement_noise(observation):
    standard_deviation = 0.35 / np.sqrt(camera_count(observation))
    return np.eye(2) * standard_deviation**2


def create_track(track_id, observation):
    return {
        "id": track_id,
        "state": np.array(
            [
                observation["world_x"],
                observation["world_y"],
                0.0,
                0.0,
            ]
        ),
        "covariance": np.diag([0.25, 0.25, 1.0, 1.0]),
        "hits": 1,
        "missed_seconds": 0.0,
        "confirmed": False,
        "visible": True,
        "history": [(observation["world_x"], observation["world_y"])],
        "embedding": observation.get("embedding"),
    }


def copy_state(track):
    track["world_x"] = float(track["state"][0])
    track["world_y"] = float(track["state"][1])
    track["velocity_x"] = float(track["state"][2])
    track["velocity_y"] = float(track["state"][3])


def predict(track, dt):
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    process_noise = np.array(
        [
            [dt4 / 4, 0.0, dt3 / 2, 0.0],
            [0.0, dt4 / 4, 0.0, dt3 / 2],
            [dt3 / 2, 0.0, dt2, 0.0],
            [0.0, dt3 / 2, 0.0, dt2],
        ]
    )
    track["state"] = transition @ track["state"]
    track["covariance"] = (
        transition @ track["covariance"] @ transition.T + process_noise
    )
    track["visible"] = False
    copy_state(track)


def mahalanobis_distance(track, observation):
    measurement = np.array(
        [observation["world_x"], observation["world_y"]]
    )
    difference = measurement - MEASUREMENT_MATRIX @ track["state"]
    innovation_covariance = (
        MEASUREMENT_MATRIX
        @ track["covariance"]
        @ MEASUREMENT_MATRIX.T
        + measurement_noise(observation)
    )
    distance_squared = float(
        difference.T @ np.linalg.inv(innovation_covariance) @ difference
    )
    return distance_squared


def cosine_distance(track, observation):
    track_embedding = track.get("embedding")
    observation_embedding = observation.get("embedding")
    if track_embedding is None or observation_embedding is None:
        return None
    return 1.0 - float(np.dot(track_embedding, observation_embedding))


def match_tracks(
    tracks,
    observations,
    gate,
    appearance_weight,
    maximum_appearance_distance,
):
    if not tracks or not observations:
        return []

    costs = np.full((len(tracks), len(observations)), 1_000_000.0)
    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            distance = mahalanobis_distance(track, observation)
            if distance > gate:
                continue

            cost = distance / gate
            appearance_distance = cosine_distance(track, observation)
            if appearance_weight > 0.0 and appearance_distance is not None:
                if appearance_distance > maximum_appearance_distance:
                    continue
                appearance_cost = (
                    appearance_distance / maximum_appearance_distance
                )
                cost = (
                    (1.0 - appearance_weight) * cost
                    + appearance_weight * appearance_cost
                )
            costs[track_index, observation_index] = cost

    rows, columns = linear_sum_assignment(costs)
    return [
        (row, column)
        for row, column in zip(rows, columns)
        if costs[row, column] < 1_000_000.0
    ]


def match_tracks_in_stages(
    tracks,
    observations,
    weak_observations,
):
    active_indices = [
        index
        for index, track in enumerate(tracks)
        if track["missed_seconds"] == 0.0
    ]
    lost_indices = [
        index
        for index, track in enumerate(tracks)
        if track["missed_seconds"] > 0.0
    ]

    active_matches = match_tracks(
        [tracks[index] for index in active_indices],
        observations,
        MAHALANOBIS_GATE,
        APPEARANCE_WEIGHT,
        MAXIMUM_APPEARANCE_DISTANCE,
    )
    strong_matches = [
        (active_indices[track_index], observation_index)
        for track_index, observation_index in active_matches
    ]

    matched_active_indices = {
        track_index for track_index, _ in strong_matches
    }
    weak_track_indices = [
        index
        for index in active_indices
        if index not in matched_active_indices and tracks[index]["confirmed"]
    ]
    local_weak_matches = match_tracks(
        [tracks[index] for index in weak_track_indices],
        weak_observations,
        MAHALANOBIS_GATE,
        APPEARANCE_WEIGHT,
        MAXIMUM_APPEARANCE_DISTANCE,
    )
    weak_matches = [
        (weak_track_indices[track_index], observation_index)
        for track_index, observation_index in local_weak_matches
    ]

    used_strong_observations = {
        observation_index for _, observation_index in strong_matches
    }
    remaining_observation_indices = [
        index
        for index in range(len(observations))
        if index not in used_strong_observations
    ]

    lost_matches = match_tracks(
        [tracks[index] for index in lost_indices],
        [observations[index] for index in remaining_observation_indices],
        MAHALANOBIS_GATE,
        REACTIVATION_APPEARANCE_WEIGHT,
        REACTIVATION_APPEARANCE_DISTANCE,
    )
    strong_matches.extend(
        (
            lost_indices[track_index],
            remaining_observation_indices[observation_index],
        )
        for track_index, observation_index in lost_matches
    )
    return strong_matches, weak_matches


def update_track(track, observation):
    measurement = np.array(
        [observation["world_x"], observation["world_y"]]
    )
    noise = measurement_noise(observation)
    innovation = measurement - MEASUREMENT_MATRIX @ track["state"]
    innovation_covariance = (
        MEASUREMENT_MATRIX
        @ track["covariance"]
        @ MEASUREMENT_MATRIX.T
        + noise
    )
    kalman_gain = (
        track["covariance"]
        @ MEASUREMENT_MATRIX.T
        @ np.linalg.inv(innovation_covariance)
    )

    track["state"] = track["state"] + kalman_gain @ innovation
    identity = np.eye(4)
    track["covariance"] = (
        identity - kalman_gain @ MEASUREMENT_MATRIX
    ) @ track["covariance"]
    track["hits"] += 1
    track["missed_seconds"] = 0.0
    track["visible"] = True
    track["confirmed"] = track["hits"] >= MINIMUM_HITS
    copy_state(track)
    track["history"].append((track["world_x"], track["world_y"]))
    track["history"] = track["history"][-HISTORY_LENGTH:]

    observation_embedding = observation.get("embedding")
    if observation_embedding is not None:
        if track.get("embedding") is None:
            track["embedding"] = observation_embedding
        else:
            embedding = (
                APPEARANCE_MOMENTUM * track["embedding"]
                + (1.0 - APPEARANCE_MOMENTUM) * observation_embedding
            )
            norm = np.linalg.norm(embedding)
            track["embedding"] = embedding / norm if norm > 0 else embedding


def attach_track_id(observation, track):
    observation["track_id"] = track["id"]
    observation["track_confirmed"] = track["confirmed"]
    for member in observation.get("members", []):
        member["track_id"] = track["id"]
        member["track_confirmed"] = track["confirmed"]
        source = member.get("source_detection")
        if source is not None:
            source["track_id"] = track["id"]
            source["track_confirmed"] = track["confirmed"]


def update_tracker(
    tracker,
    observations,
    dt,
    weak_observations,
):
    tracks = tracker["tracks"]
    for track in tracks:
        predict(track, dt)

    strong_matches, weak_matches = match_tracks_in_stages(
        tracks,
        observations,
        weak_observations,
    )
    matched_tracks = {
        track_index for track_index, _ in strong_matches + weak_matches
    }
    matched_observations = {
        observation_index for _, observation_index in strong_matches
    }

    matched_pairs = [
        (track_index, observations[observation_index])
        for track_index, observation_index in strong_matches
    ]
    matched_pairs.extend(
        (track_index, weak_observations[observation_index])
        for track_index, observation_index in weak_matches
    )
    for track_index, observation in matched_pairs:
        track = tracks[track_index]
        update_track(track, observation)
        attach_track_id(observation, track)
        if track["confirmed"]:
            tracker["confirmed_ids"].add(track["id"])

    for track_index, track in enumerate(tracks):
        if track_index not in matched_tracks:
            track["missed_seconds"] += dt

    tracker["tracks"] = [
        track
        for track in tracks
        if track["missed_seconds"] <= MAXIMUM_MISSED_SECONDS
    ]

    for observation_index, observation in enumerate(observations):
        if observation_index in matched_observations:
            continue
        track = create_track(tracker["next_id"], observation)
        copy_state(track)
        tracker["tracks"].append(track)
        attach_track_id(observation, track)
        tracker["next_id"] += 1
        tracker["created_ids"] += 1

    return tracker["tracks"]
