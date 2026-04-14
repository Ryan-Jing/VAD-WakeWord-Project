#==================================================================================================#
# @name dtw.py
#
# @brief    Dynamic Time Warping implementation for keyword spotting
#==================================================================================================#

import numpy as np

from config import Config

class DTW:
    def __init__(self, keyword_template, distance_threshold = None):
        self.config = Config()

        self.keyword_template = keyword_template
        self.distance_threshold = self.config.dtw_distance_threshold if distance_threshold is None else distance_threshold

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def _backtrack_warping_path(self, accumulated_cost_matrix):
        template_index = accumulated_cost_matrix.shape[0] - 1
        feature_index = accumulated_cost_matrix.shape[1] - 1

        warping_path = [(template_index, feature_index)]

        while template_index > 0 or feature_index > 0:
            previous_steps = []

            if template_index > 0:
                previous_steps.append((accumulated_cost_matrix[template_index - 1, feature_index], template_index - 1, feature_index))
            if feature_index > 0:
                previous_steps.append((accumulated_cost_matrix[template_index, feature_index - 1], template_index, feature_index - 1))
            if template_index > 0 and feature_index > 0:
                previous_steps.append((accumulated_cost_matrix[template_index - 1, feature_index - 1], template_index - 1, feature_index - 1))

            _, template_index, feature_index = min(previous_steps, key=lambda step: step[0])
            warping_path.append((template_index, feature_index))

        warping_path.reverse()
        return np.array(warping_path)

    def compute_dtw_analysis(self, features):
        template = self.keyword_template
        template_frames = template.shape[0]
        feature_frames = features.shape[0]

        if template_frames == 0 or feature_frames == 0:
            raise ValueError("DTW requires non-empty template and feature sequences.")

        # Sakoe-Chiba band width, ±20% of the longer sequence
        band_width = max(1, int(0.2 * max(template_frames, feature_frames)))

        frame_distance_matrix = np.empty((template_frames, feature_frames), dtype=np.float64)
        for template_index in range(template_frames):
            for feature_index in range(feature_frames):
                frame_distance_matrix[template_index, feature_index] = np.linalg.norm(
                    template[template_index] - features[feature_index]
                )

        accumulated_cost_matrix = np.full((template_frames, feature_frames), np.inf, dtype=np.float64)

        for template_index in range(template_frames):
            feature_start = max(0, template_index - band_width)
            feature_end = min(feature_frames, template_index + band_width + 1)

            for feature_index in range(feature_start, feature_end):
                local_cost = frame_distance_matrix[template_index, feature_index]

                if template_index == 0 and feature_index == 0:
                    accumulated_cost_matrix[template_index, feature_index] = local_cost
                    continue

                insertion_cost = accumulated_cost_matrix[template_index - 1, feature_index] if template_index > 0 else np.inf
                deletion_cost = accumulated_cost_matrix[template_index, feature_index - 1] if feature_index > 0 else np.inf
                match_cost = accumulated_cost_matrix[template_index - 1, feature_index - 1] if template_index > 0 and feature_index > 0 else np.inf

                accumulated_cost_matrix[template_index, feature_index] = local_cost + min(
                    insertion_cost,
                    deletion_cost,
                    match_cost
                )

        warping_path = self._backtrack_warping_path(accumulated_cost_matrix)
        normalized_dtw_distance = accumulated_cost_matrix[template_frames - 1, feature_frames - 1] / (template_frames + feature_frames)

        return {
            "dtw_distance": normalized_dtw_distance,
            "frame_distance_matrix": frame_distance_matrix,
            "accumulated_cost_matrix": accumulated_cost_matrix,
            "warping_path": warping_path,
            "sakoe_chiba_band_width": band_width
        }

    def compute_dtw_distance(self, features):
        return self.compute_dtw_analysis(features)["dtw_distance"]

    def is_keyword_detected(self, features, return_analysis = False):
        dtw_analysis = self.compute_dtw_analysis(features)
        detected = dtw_analysis["dtw_distance"] < self.distance_threshold

        if return_analysis:
            return detected, dtw_analysis["dtw_distance"], dtw_analysis

        return detected, dtw_analysis["dtw_distance"]
