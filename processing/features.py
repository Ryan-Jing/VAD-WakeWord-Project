#==================================================================================================#
# @name features.py
#
# @brief   Extraction of features from audio data for keyword spotting
#
# @to-do:
#  - Get the length of keyword based on the keyword audio recordings, instead of setting the
#    keyword frame length to a fixed value. The calibration sequence should record and stop recording
#    based on voice activity detection, and the keyword frame length should be determined based on the
#    length of the detected keyword audio segment.
#
#    Then, when we want to only extract features after a certain number of time of detected speecj,
#    we use the keyword frame length we generated to say, if speech is detected for 70% of the
#    keyword frame length, then we start extracting features and checking for keyword detection.
#==================================================================================================#

import numpy as np

from scipy.fft import dct
from config import Config

class Features:
    def __init__(self):
        self.config = Config()

        self.sampling_rate_hz = self.config.sampling_rate_hz
        self.frame_length = self.config.frame_length
        self.hop_length = self.config.hop_length
        self.smooth_window_length = 5
        self.keyword_frame_length = 0
        self.feature_mean = None
        self.feature_std = None

    def set_keyword_frame_length(self, length):
        self.keyword_frame_length = length

    def get_keyword_frame_length(self):
        return self.keyword_frame_length

    def fit_normalization(self, feature_segments):
        if len(feature_segments) == 0:
            self.feature_mean = None
            self.feature_std = None
            return

        stacked_features = np.vstack(feature_segments)
        self.feature_mean = np.mean(stacked_features, axis=0)
        self.feature_std = np.std(stacked_features, axis=0) + 1e-12

    def normalize_features(self, features):
        if features.size == 0:
            return features

        if self.feature_mean is None or self.feature_std is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-12
            return (features - mean) / std

        return (features - self.feature_mean) / self.feature_std

    def extract_features(self, audio_segment, normalize = True):
        if self.config.project_testing:
            print("Extracting features...")

        audio_segment_spectrum = self._stft(audio_segment)
        audio_segment_mfcc = self._mfcc(audio_segment_spectrum)
        deltas = self._get_deltas(audio_segment_mfcc)
        delta_deltas = self._get_deltas(deltas)
        features = np.concatenate([audio_segment_mfcc, deltas, delta_deltas], axis = 1)

        if normalize:
            return self.normalize_features(features)

        return features

    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _stft(self, audio_segment):
        if len(audio_segment) < self.frame_length:
            audio_segment = np.pad(audio_segment, (0, self.frame_length - len(audio_segment)))

        audio_frames_spectrum = []
        for start in range(0, len(audio_segment) - self.frame_length + 1, self.hop_length):
            audio_frame = audio_segment[start : start + self.frame_length]
            audio_frame_hamming = audio_frame * np.hamming(self.frame_length)
            audio_frame_spectrum = np.abs(np.fft.rfft(audio_frame_hamming, n = 512)) ** 2
            audio_frames_spectrum.append(audio_frame_spectrum)
        return np.array(audio_frames_spectrum)

    def _mfcc(self, audio_segment_spectrum):
        mel_filters = self._get_mel_filters(num_filters=40, n_fft=512)
        mel_spectrum = np.dot(audio_segment_spectrum, mel_filters.T)
        log_mel = np.log(mel_spectrum + 1e-12)

        # DCT to get MFCCs — keep 13 coefficients
        mfccs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :13] # type: ignore
        return mfccs

    def _get_deltas(self, features, width = 2):
        deltas = np.zeros_like(features)
        for t in range(width, features.shape[0] - width):
            numerator = sum(n * (features[t + n] - features[t - n]) for n in range(1, width + 1))
            denominator = 2 * sum(n ** 2 for n in range(1, width + 1))
            deltas[t] = numerator / denominator
        return deltas

    def _get_mel_filters(self, num_filters, n_fft):
        mel_low_freq = self._hz_to_mel(self.config.speech_low_cutoff_hz)
        mel_high_freq = self._hz_to_mel(self.config.speech_high_cutoff_hz)
        mel_points = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sampling_rate_hz).astype(int)

        mel_filters = np.zeros((num_filters, n_fft // 2 + 1))

        for i in range(num_filters):
            for j in range(bin_points[i], bin_points[i + 1]):
                mel_filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                mel_filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

        return mel_filters
