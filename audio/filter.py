#========================================================================#
# @name filter.py
#
# @brief   Helper functions to filter audio data
#========================================================================#

import numpy as np
from scipy.signal import butter, filtfilt, sosfilt, zpk2tf

from config import Config

class AudioFilter:
    def __init__(self):
        self.config = Config()
        self.sampling_rate_hz = self.config.sampling_rate_hz
        self.notch_filter_hz = self.config.notch_filter_hz
        self.preemphasis_coeff = 0.97
        self.dc_blocking_pole = 0.995

        nyquist = 0.5 * self.sampling_rate_hz
        low = self.config.speech_low_cutoff_hz / nyquist
        high = self.config.speech_high_cutoff_hz / nyquist
        self.bandpass_sos = butter(4, [low, high], btype='band', output='sos') #type: ignore

        self.reset_streaming_state()

    def reset_streaming_state(self):
        self._bandpass_state = np.zeros((self.bandpass_sos.shape[0], 2), dtype=np.float64) #type: ignore
        self._dc_previous_input = 0.0
        self._dc_previous_output = 0.0
        self._preemphasis_previous_sample = 0.0

    def filter_audio(self, data):
        self.reset_streaming_state()
        return self.filter_audio_chunk(data)

    def filter_audio_chunk(self, data):
        if len(data) == 0:
            return np.array([], dtype=np.float32)

        streaming_audio = np.asarray(data, dtype=np.float64)

        dc_filtered_data = self._dc_block_filter_chunk(streaming_audio)
        bw_hp_filtered_data, self._bandpass_state = sosfilt(self.bandpass_sos, dc_filtered_data, zi=self._bandpass_state)
        preemphasized_data = self._preemphasize_chunk(bw_hp_filtered_data)

        return preemphasized_data.astype(np.float32)

    def _dc_block_filter_chunk(self, data):
        dc_filtered_data = np.zeros_like(data)

        previous_input = self._dc_previous_input
        previous_output = self._dc_previous_output

        for index, sample in enumerate(data):
            current_output = sample - previous_input + self.dc_blocking_pole * previous_output
            dc_filtered_data[index] = current_output

            previous_input = sample
            previous_output = current_output

        self._dc_previous_input = float(previous_input)
        self._dc_previous_output = float(previous_output)

        return dc_filtered_data

    def _preemphasize_chunk(self, data):
        preemphasized_data = np.empty_like(data)

        preemphasized_data[0] = data[0] - self.preemphasis_coeff * self._preemphasis_previous_sample

        if len(data) > 1:
            preemphasized_data[1:] = data[1:] - self.preemphasis_coeff * data[:-1]

        self._preemphasis_previous_sample = float(data[-1])

        return preemphasized_data

    def dc_offset_filter(self, data):
        dc_filtered_data = data - np.mean(data)
        return dc_filtered_data

    def butterworth_bandpass_filter(self, data, order = 4):
        nyquist = 0.5 * self.sampling_rate_hz
        low = self.config.speech_low_cutoff_hz / nyquist
        high = self.config.speech_high_cutoff_hz / nyquist
        b, a = butter(order, [low, high], btype='band') #type: ignore
        bw_hp_filtered_data = filtfilt(b, a, data)
        return bw_hp_filtered_data

    def preemphasize_filter(self, data, preemphasis_coeff = 0.97):
        preemphasized_data = np.append(data[0], data[1:] - preemphasis_coeff * data[:-1])
        return preemphasized_data

    # The following functions are if needed, but I don't think they are necessary for our project
    def butterworth_lowpass_filter(self, data, cutoff, order = 4):
        nyquist = 0.5 * self.sampling_rate_hz
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low') #type: ignore
        bw_lp_filtered_data = filtfilt(b, a, data)
        return bw_lp_filtered_data

    def notch_filter(self, data):
        theta = 2 * np.pi * self.notch_filter_hz / self.sampling_rate_hz
        z_1 = np.exp(-1j * theta)
        z_1_conjugate = np.conj(z_1)
        zeroes_vector = [z_1, z_1_conjugate]

        pole_radius = 0.98
        pole = pole_radius * np.exp(-1j * theta)
        pole_conjugate = np.conj(pole)
        poles_vector = [pole, pole_conjugate]

        notch_filter = zpk2tf(zeroes_vector, poles_vector, k = 1)
        notch_filtered_data = filtfilt(notch_filter[0], notch_filter[1], data)
        return notch_filtered_data

    def rectify_signal(self, data):
        return np.abs(data)