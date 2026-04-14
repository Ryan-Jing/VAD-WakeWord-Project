#==================================================================================================#
# @name config.py
#
# @brief Configs for the project
#==================================================================================================#

from enum import Enum

class ProjectConfig(Enum):
    PRODUCTION = 0
    TESTING = 1

class Config:
    def __init__(self):
        self.sampling_rate_hz = 16000
        self.speech_low_cutoff_hz = 80
        self.speech_high_cutoff_hz = 7000
        self.notch_filter_hz = 60

        self.frame_length = 400
        self.hop_length = 160
        self.smooth_window_length = 5
        self.keyword_frame_length = 0
        self.project_testing = ProjectConfig

        self.calibration_keyword_duration_secs = 3
        self.calibration_silence_duration_secs = 3
        self.calibration_num_iterations = 3

        self.dtw_distance_threshold = 3.0
