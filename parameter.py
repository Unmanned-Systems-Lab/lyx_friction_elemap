#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from dataclasses import dataclass
import pickle
import numpy as np
import os


@dataclass
class Parameter:
    resolution: float = 0.04

    map_length: float = 20.0
    sensor_noise_factor: float = 0.05
    mahalanobis_thresh: float = 2
    outlier_variance: float = 0.01
    drift_compensation_variance_inlier: float = 0.05
    time_variance: float = 0.0001
    time_interval: float = 0.1

    max_variance: float = 100.0
    dilation_size: float = 3
    dilation_size_initialize: float = 5
    drift_compensation_alpha: float = 0.1

    traversability_inlier: float = 0.9
    wall_num_thresh: float = 20
    min_height_drift_cnt: float = 100

    max_ray_length: float = 10.0
    cleanup_step: float = 0.1
    cleanup_cos_thresh: float = 0.1
    min_valid_distance: float = 0.5
    # TODO ht
    max_height_range: float = 2.0
    ramped_height_range_a: float = 0
    ramped_height_range_b: float = 1.0
    ramped_height_range_c: float = 2

    safe_thresh: float = 0.7
    safe_min_thresh: float = 0.4
    max_unsafe_n: int = 10

    min_filter_size: int = 5
    min_filter_iteration: int = 3

    max_drift: float = 0.10

    # TODO ht
    overlap_clear_range_xy: float = 4.0
    overlap_clear_range_z: float = 2.0

    enable_edge_sharpen: bool = True
    enable_drift_compensation: bool = True
    enable_visibility_cleanup: bool = True
    enable_overlap_clearance: bool = False
    use_only_above_for_upper_bound: bool = False
    use_chainer: bool = False
    position_noise_thresh: float = 0.01
    orientation_noise_thresh: float = 0.01

    plugin_config_file: str = "config/plugin_config.yaml"
    weight_file: str = "config/weights.dat"

    initial_variance: float = 1000.0
    initialized_variance: float = 10.0
    w1: np.ndarray = np.zeros((4, 1, 3, 3))
    w2: np.ndarray = np.zeros((4, 1, 3, 3))
    w3: np.ndarray = np.zeros((4, 1, 3, 3))
    w_out: np.ndarray = np.zeros((1, 12, 1, 1))

    def load_weights(self, filename):
        with open(filename, "rb") as file:
            weights = pickle.load(file)
            self.w1 = weights["conv1.weight"]
            self.w2 = weights["conv2.weight"]
            self.w3 = weights["conv3.weight"]
            self.w_out = weights["conv_final.weight"]

    def get_names(self):
        return list(self.__annotations__.keys())

    def get_types(self):
        return [v.__name__ for v in self.__annotations__.values()]

    def set_value(self, name, value):
        setattr(self, name, value)

    def get_value(self, name):
        return getattr(self, name)


if __name__ == "__main__":
    param = Parameter()
    print(param)
    print(param.resolution)
    param.set_value("resolution", 0.1)
    print(param.resolution)

    print("names ", param.get_names())
    print("types ", param.get_types())
