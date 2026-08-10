import math
import os
import xml.etree.ElementTree as ET

import numpy as np


def create_rotation_matrix(rotation_x, rotation_y, rotation_z):
    sin_x = math.sin(rotation_x)
    cos_x = math.cos(rotation_x)
    sin_y = math.sin(rotation_y)
    cos_y = math.cos(rotation_y)
    sin_z = math.sin(rotation_z)
    cos_z = math.cos(rotation_z)

    return np.array(
        [
            [
                cos_y * cos_z,
                cos_z * sin_x * sin_y - cos_x * sin_z,
                sin_x * sin_z + cos_x * cos_z * sin_y,
            ],
            [
                cos_y * sin_z,
                sin_x * sin_y * sin_z + cos_x * cos_z,
                cos_x * sin_y * sin_z - cos_z * sin_x,
            ],
            [-sin_y, cos_y * sin_x, cos_x * cos_y],
        ],
        dtype=float,
    )


def load_camera_calibration(dataset_path, camera_index):
    calibration_path = os.path.join(
        dataset_path,
        "calibration",
        "tsai",
        "terrace-tsai-c" + str(camera_index) + ".xml",
    )
    root = ET.parse(calibration_path).getroot()
    geometry = root.find("Geometry").attrib
    intrinsic = root.find("Intrinsic").attrib
    extrinsic = root.find("Extrinsic").attrib

    rotation = create_rotation_matrix(
        float(extrinsic["rx"]),
        float(extrinsic["ry"]),
        float(extrinsic["rz"]),
    )
    translation = np.array(
        [
            float(extrinsic["tx"]),
            float(extrinsic["ty"]),
            float(extrinsic["tz"]),
        ]
    )

    return {
        "calibration_width": float(geometry["width"]),
        "calibration_height": float(geometry["height"]),
        "pixel_size_x": float(geometry["dpx"]),
        "pixel_size_y": float(geometry["dpy"]),
        "focal_length": float(intrinsic["focal"]),
        "radial_distortion": float(intrinsic["kappa1"]),
        "center_x": float(intrinsic["cx"]),
        "center_y": float(intrinsic["cy"]),
        "scale_x": float(intrinsic["sx"]),
        "rotation": rotation,
        "translation": translation,
    }


def load_camera_calibrations(dataset_path):
    return [
        load_camera_calibration(dataset_path, camera_index)
        for camera_index in range(4)
    ]


def undistorted_to_distorted(undistorted_x, undistorted_y, distortion):
    undistorted_radius = math.hypot(undistorted_x, undistorted_y)
    if undistorted_radius == 0:
        return undistorted_x, undistorted_y

    distorted_radius = undistorted_radius
    for _ in range(10):
        function = (
            distorted_radius
            + distortion * distorted_radius**3
            - undistorted_radius
        )
        derivative = 1.0 + 3.0 * distortion * distorted_radius**2
        distorted_radius -= function / derivative

    scale = distorted_radius / undistorted_radius
    return undistorted_x * scale, undistorted_y * scale


def world_to_image(
    world_x,
    world_y,
    world_z,
    calibration,
    image_width,
    image_height,
):
    # Terrace world coordinates and Tsai translations are in millimetres.
    world_point = np.array(
        [world_x * 1000.0, world_y * 1000.0, world_z * 1000.0]
    )
    camera_point = (
        calibration["rotation"] @ world_point
        + calibration["translation"]
    )
    if camera_point[2] <= 0:
        return None

    undistorted_x = (
        calibration["focal_length"] * camera_point[0] / camera_point[2]
    )
    undistorted_y = (
        calibration["focal_length"] * camera_point[1] / camera_point[2]
    )
    distorted_x, distorted_y = undistorted_to_distorted(
        undistorted_x,
        undistorted_y,
        calibration["radial_distortion"],
    )

    calibration_x = (
        distorted_x
        * calibration["scale_x"]
        / calibration["pixel_size_x"]
        + calibration["center_x"]
    )
    calibration_y = (
        distorted_y / calibration["pixel_size_y"]
        + calibration["center_y"]
    )

    image_x = (
        calibration_x * image_width / calibration["calibration_width"]
    )
    image_y = (
        calibration_y * image_height / calibration["calibration_height"]
    )
    return float(image_x), float(image_y)


def image_to_world(
    image_x,
    image_y,
    calibration,
    image_width,
    image_height,
):
    calibration_x = (
        image_x * calibration["calibration_width"] / image_width
    )
    calibration_y = (
        image_y * calibration["calibration_height"] / image_height
    )

    distorted_x = (
        (calibration_x - calibration["center_x"])
        * calibration["pixel_size_x"]
        / calibration["scale_x"]
    )
    distorted_y = (
        (calibration_y - calibration["center_y"])
        * calibration["pixel_size_y"]
    )
    radius_squared = distorted_x**2 + distorted_y**2
    distortion_scale = (
        1.0 + calibration["radial_distortion"] * radius_squared
    )
    undistorted_x = distorted_x * distortion_scale
    undistorted_y = distorted_y * distortion_scale

    camera_ray = np.array(
        [
            undistorted_x / calibration["focal_length"],
            undistorted_y / calibration["focal_length"],
            1.0,
        ]
    )
    inverse_rotation = calibration["rotation"].T
    numerator = (inverse_rotation @ calibration["translation"])[2]
    denominator = (inverse_rotation @ camera_ray)[2]
    if abs(denominator) < 1e-12:
        return None

    ray_scale = numerator / denominator
    world_point = inverse_rotation @ (
        ray_scale * camera_ray - calibration["translation"]
    )
    return float(world_point[0] / 1000.0), float(world_point[1] / 1000.0)


def project_box_foot(box, calibration, image_width, image_height):
    foot_x = (box[0] + box[2]) / 2.0
    foot_y = box[3]
    return image_to_world(
        foot_x,
        foot_y,
        calibration,
        image_width,
        image_height,
    )


def calculate_round_trip_errors(people, calibrations, image_width, image_height):
    errors = []
    for calibration in calibrations:
        for person in people:
            image_point = world_to_image(
                person["world_x"],
                person["world_y"],
                0.0,
                calibration,
                image_width,
                image_height,
            )
            if image_point is None:
                continue

            recovered_point = image_to_world(
                image_point[0],
                image_point[1],
                calibration,
                image_width,
                image_height,
            )
            if recovered_point is None:
                continue

            error = math.hypot(
                recovered_point[0] - person["world_x"],
                recovered_point[1] - person["world_y"],
            )
            errors.append(error)
    return errors
