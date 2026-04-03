#========================================================================#
# @name filter.py
#
# @brief   Helper functions to filter audio data
#========================================================================#

import numpy as np
from scipy.signal import butter, lfilter, filtfilt, zpk2tf

class AudioFilter:
    def __init__(self, sampling_rate_hz = 16000, notch_filter_hz = 60):
        self.sampling_rate_hz = sampling_rate_hz
        self.notch_filter_hz = notch_filter_hz

    def dc_offset_filter(self, data):
        dc_filtered_data = data - np.mean(data)
        return dc_filtered_data

    def butterworth_bandpass_filter(self, data, low_cutoff = 80, high_cutoff = 4000, order = 4):
        nyquist = 0.5 * self.sampling_rate_hz
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
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

