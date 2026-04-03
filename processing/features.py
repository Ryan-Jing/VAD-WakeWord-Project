#==================================================================================================#
# @name features.py
#
# @brief   Extraction of features from audio data for keyword spotting
#==================================================================================================#

import numpy as np

from config import Config

class Features:
    def __init__(self):
        self.keyword_frame_length = 0
        self.config = Config
        pass

    def get_keyword_frame_length(self):
        if self.config.TESTING:
            self.keyword_frame_length = 4
        return self.keyword_frame_length