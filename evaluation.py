import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def create_evaluator():
    return {
        "frames": [],
        "ground_truth_total": 0,
        "prediction_total": 0,
        "true_positives": 0,
        "errors": [],
        "id_switches": 0,
        "previous_matches": {},
        "identity_matches": defaultdict(int),
        "person_match_history": defaultdict(list),
        "matching_distance": None,
        "hota_frames": [],
    }


def evaluate_frame(evaluator, frame_index, people, tracks, matching_distance):
    evaluator["matching_distance"] = matching_distance
    predictions = [
        track
        for track in tracks
        if track["confirmed"] and track["visible"]
    ]
    evaluator["ground_truth_total"] += len(people)
    evaluator["prediction_total"] += len(predictions)

    ground_truth_points = np.array(
        [[person["world_x"], person["world_y"]] for person in people]
    ).reshape(-1, 2)
    predicted_points = np.array(
        [[track["world_x"], track["world_y"]] for track in predictions]
    ).reshape(-1, 2)
    distances = np.zeros((len(people), len(predictions)))
    if people and predictions:
        distances = np.linalg.norm(
            ground_truth_points[:, None, :] - predicted_points[None, :, :],
            axis=2,
        )

    similarity = np.clip(1.0 - distances / matching_distance, 0.0, 1.0)
    evaluator["hota_frames"].append(
        {
            "ground_truth_ids": [person["person_id"] for person in people],
            "tracker_ids": [track["id"] for track in predictions],
            "similarity": similarity,
        }
    )

    matches = []
    if people and predictions:
        reserved_people = set()
        reserved_predictions = set()
        prediction_by_id = {
            track["id"]: index for index, track in enumerate(predictions)
        }

        # CLEAR MOT first preserves valid correspondences from the previous
        # evaluated timestamp, then assigns the remaining objects.
        for person_index, person in enumerate(people):
            previous_track_id = evaluator["previous_matches"].get(
                person["person_id"]
            )
            prediction_index = prediction_by_id.get(previous_track_id)
            if prediction_index is None:
                continue
            distance = float(distances[person_index, prediction_index])
            if distance <= matching_distance:
                matches.append((person_index, prediction_index, distance))
                reserved_people.add(person_index)
                reserved_predictions.add(prediction_index)

        remaining_people = [
            index for index in range(len(people)) if index not in reserved_people
        ]
        remaining_predictions = [
            index
            for index in range(len(predictions))
            if index not in reserved_predictions
        ]
        if remaining_people and remaining_predictions:
            remaining_distances = distances[
                np.ix_(remaining_people, remaining_predictions)
            ]
            rows, columns = linear_sum_assignment(remaining_distances)
            for row, column in zip(rows, columns):
                distance = float(remaining_distances[row, column])
                if distance <= matching_distance:
                    matches.append(
                        (
                            remaining_people[row],
                            remaining_predictions[column],
                            distance,
                        )
                    )

    frame_switches = 0
    current_matches = {}
    matched_person_ids = set()
    for person_index, track_index, error in matches:
        person_id = people[person_index]["person_id"]
        track_id = predictions[track_index]["id"]
        previous_track_id = evaluator["previous_matches"].get(person_id)
        if previous_track_id is not None and previous_track_id != track_id:
            frame_switches += 1
        current_matches[person_id] = track_id
        matched_person_ids.add(person_id)
        evaluator["identity_matches"][(person_id, track_id)] += 1
        evaluator["errors"].append(error)

    for person in people:
        evaluator["person_match_history"][person["person_id"]].append(
            person["person_id"] in matched_person_ids
        )
    evaluator["previous_matches"] = current_matches

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(people) - true_positives
    evaluator["true_positives"] += true_positives
    evaluator["id_switches"] += frame_switches
    evaluator["frames"].append(
        {
            "frame": frame_index,
            "ground_truth": len(people),
            "predictions": len(predictions),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "id_switches": frame_switches,
            "mean_error": (
                sum(error for _, _, error in matches) / true_positives
                if true_positives
                else 0.0
            ),
        }
    )


def calculate_idtp(evaluator):
    pairs = evaluator["identity_matches"]
    if not pairs:
        return 0

    person_ids = sorted({person_id for person_id, _ in pairs})
    track_ids = sorted({track_id for _, track_id in pairs})
    counts = np.zeros((len(person_ids), len(track_ids)), dtype=int)
    person_rows = {person_id: row for row, person_id in enumerate(person_ids)}
    track_columns = {
        track_id: column for column, track_id in enumerate(track_ids)
    }
    for (person_id, track_id), count in pairs.items():
        counts[person_rows[person_id], track_columns[track_id]] = count

    rows, columns = linear_sum_assignment(-counts)
    return int(counts[rows, columns].sum())


def calculate_hota(evaluator):
    """Calculate TrackEval-style HOTA using ground-plane distance similarity."""
    frames = evaluator["hota_frames"]
    thresholds = np.arange(0.05, 1.0, 0.05)
    ground_truth_ids = sorted(
        {
            person_id
            for frame in frames
            for person_id in frame["ground_truth_ids"]
        }
    )
    tracker_ids = sorted(
        {
            track_id
            for frame in frames
            for track_id in frame["tracker_ids"]
        }
    )
    ground_truth_rows = {
        person_id: row for row, person_id in enumerate(ground_truth_ids)
    }
    tracker_columns = {
        track_id: column for column, track_id in enumerate(tracker_ids)
    }

    pair_shape = (len(ground_truth_ids), len(tracker_ids))
    potential_matches = np.zeros(pair_shape)
    ground_truth_counts = np.zeros((len(ground_truth_ids), 1))
    tracker_counts = np.zeros((1, len(tracker_ids)))

    prepared_frames = []
    for frame in frames:
        frame_ground_truth = np.array(
            [ground_truth_rows[value] for value in frame["ground_truth_ids"]],
            dtype=int,
        )
        frame_trackers = np.array(
            [tracker_columns[value] for value in frame["tracker_ids"]],
            dtype=int,
        )
        similarity = frame["similarity"]
        prepared_frames.append(
            (frame_ground_truth, frame_trackers, similarity)
        )

        if len(frame_ground_truth) and len(frame_trackers):
            denominator = (
                similarity.sum(axis=0)[None, :]
                + similarity.sum(axis=1)[:, None]
                - similarity
            )
            normalized_similarity = np.divide(
                similarity,
                denominator,
                out=np.zeros_like(similarity),
                where=denominator > np.finfo(float).eps,
            )
            potential_matches[np.ix_(frame_ground_truth, frame_trackers)] += (
                normalized_similarity
            )
        ground_truth_counts[frame_ground_truth, 0] += 1
        tracker_counts[0, frame_trackers] += 1

    alignment_denominator = (
        ground_truth_counts + tracker_counts - potential_matches
    )
    global_alignment = np.divide(
        potential_matches,
        alignment_denominator,
        out=np.zeros_like(potential_matches),
        where=alignment_denominator > np.finfo(float).eps,
    )

    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    false_negatives = np.zeros(len(thresholds))
    match_counts = [np.zeros(pair_shape) for _ in thresholds]

    for frame_ground_truth, frame_trackers, similarity in prepared_frames:
        if not len(frame_ground_truth):
            false_positives += len(frame_trackers)
            continue
        if not len(frame_trackers):
            false_negatives += len(frame_ground_truth)
            continue

        alignment = global_alignment[
            np.ix_(frame_ground_truth, frame_trackers)
        ]
        rows, columns = linear_sum_assignment(-(alignment * similarity))
        for threshold_index, threshold in enumerate(thresholds):
            valid = similarity[rows, columns] >= threshold - np.finfo(float).eps
            matched_rows = rows[valid]
            matched_columns = columns[valid]
            match_count = len(matched_rows)
            true_positives[threshold_index] += match_count
            false_negatives[threshold_index] += (
                len(frame_ground_truth) - match_count
            )
            false_positives[threshold_index] += (
                len(frame_trackers) - match_count
            )
            if match_count:
                pairs = np.ix_(
                    frame_ground_truth[matched_rows],
                    frame_trackers[matched_columns],
                )
                match_counts[threshold_index][pairs] += np.eye(match_count)

    association_accuracy = np.zeros(len(thresholds))
    for threshold_index, counts in enumerate(match_counts):
        association = counts / np.maximum(
            1.0,
            ground_truth_counts + tracker_counts - counts,
        )
        denominator = max(1.0, true_positives[threshold_index])
        association_accuracy[threshold_index] = (
            counts * association
        ).sum() / denominator

    detection_accuracy = true_positives / np.maximum(
        1.0, true_positives + false_negatives + false_positives
    )
    hota = np.sqrt(detection_accuracy * association_accuracy)

    return {
        "hota": float(hota.mean()),
        "deta": float(detection_accuracy.mean()),
        "assa": float(association_accuracy.mean()),
    }


def summarize_evaluator(evaluator):
    ground_truth_total = evaluator["ground_truth_total"]
    prediction_total = evaluator["prediction_total"]
    true_positives = evaluator["true_positives"]
    false_positives = prediction_total - true_positives
    false_negatives = ground_truth_total - true_positives
    mota = (
        1.0
        - (false_negatives + false_positives + evaluator["id_switches"])
        / ground_truth_total
        if ground_truth_total
        else 0.0
    )
    idtp = calculate_idtp(evaluator)
    idf1_denominator = ground_truth_total + prediction_total
    idf1 = 2 * idtp / idf1_denominator if idf1_denominator else 0.0

    fragmentations = 0
    for history in evaluator["person_match_history"].values():
        matched_segments = 0
        previously_matched = False
        for matched in history:
            if matched and not previously_matched:
                matched_segments += 1
            previously_matched = matched
        fragmentations += max(0, matched_segments - 1)

    mean_error = (
        float(np.mean(evaluator["errors"])) if evaluator["errors"] else 0.0
    )

    summary = {
        "matching_distance_m": evaluator["matching_distance"],
        "evaluated_frames": len(evaluator["frames"]),
        "ground_truth": ground_truth_total,
        "predictions": prediction_total,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "motp_cm": mean_error * 100.0,
        "id_switches": evaluator["id_switches"],
        "fragmentations": fragmentations,
        "mota": mota,
        "idf1": idf1,
    }
    summary.update(calculate_hota(evaluator))
    return summary


def save_evaluation(evaluator, summary, summary_path):
    with open(summary_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    frame_path = summary_path.rsplit(".", 1)[0] + "_frames.csv"
    frame_fields = [
        "frame",
        "ground_truth",
        "predictions",
        "true_positives",
        "false_positives",
        "false_negatives",
        "id_switches",
        "mean_error",
    ]
    with open(frame_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=frame_fields,
        )
        writer.writeheader()
        writer.writerows(evaluator["frames"])
    return frame_path
