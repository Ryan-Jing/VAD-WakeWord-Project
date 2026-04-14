#==================================================================================================#
# @name calibration.py
#
# @brief    Calibration and training of keyword spotting pipeline
#==================================================================================================#

import time
import numpy as np

from scipy.io.wavfile import write

from audio.capture import AudioStream
from audio.filter import AudioFilter
from audio.capture import KeywordAudioSetup
from processing.vad import VoiceActivityDetection
from processing.features import Features
from config import Config
from utils import Utils

class Calibration:
    def __init__(self):
        self.config = Config()
        self.audio_stream = AudioStream()
        self.audio_filter = AudioFilter()
        self.calibration = KeywordAudioSetup()
        self.vad = VoiceActivityDetection()
        self.features = Features()
        self.utils = Utils()

        self.filtered_keyword_recordings = []
        self.keyword_template = None
        self.keyword_frame_length = 0

        self.audio_recorded = False

    def get_keyword_template(self):
        return self.keyword_template

    def get_dtw_distance_threshold(self):
        return self.config.dtw_distance_threshold

    def calibration_sequence(self):
        print("Recording silence for calibration, please be quiet")
        silence_audio = self.calibration.record_silence(self.config.calibration_silence_duration_secs)
        silence_filtered = self.audio_filter.filter_audio(silence_audio)
        self.vad.vad_calibration(silence_filtered)

        for number in range(self.config.calibration_num_iterations):
            print(f"Recording keyword for calibration, please say the keyword ({number + 1}/{self.config.calibration_num_iterations})...")
            raw_audio = self.calibration.record_utterance(self.config.calibration_keyword_duration_secs)

            filtered_audio = self.audio_filter.filter_audio(raw_audio)
            isolated_speech = self._isolate_speech_segment(filtered_audio)

            if self.config.project_testing and isolated_speech is not None:
                audio_int16 = (isolated_speech * 32767).astype(np.int16)
                write(f"recordings/keyword_{number}.wav", self.config.sampling_rate_hz, audio_int16)
                write(filename=f"recordings/raw_keyword_{number}.wav", rate=self.config.sampling_rate_hz, data=(raw_audio * 32767).astype(np.int16))
                self.utils.plot_audio_data(isolated_speech, title=f"Isolated Speech Recording {number + 1}")
                self.utils.plot_audio_data(raw_audio, title=f"Raw Audio Recording {number + 1}")

            if isolated_speech is not None:
                self.filtered_keyword_recordings.append(isolated_speech)
            else:
                print(f"No speech detected in recording {number + 1}")

            print("Finished recording keyword")
            time.sleep(1)

        self.keyword_frame_length = np.mean([len(recording) for recording in self.filtered_keyword_recordings]) / self.config.hop_length
        self.features.set_keyword_frame_length(self.keyword_frame_length)
        self.audio_recorded = True
        self._characterize_keyword_audio()
        self.vad.reset_state()

    def _build_keyword_template(self, features):
        feature_lengths = [feature.shape[0] for feature in features]
        target_length = int(np.median(feature_lengths))

        resampled_features = []
        for feature in features:
            if self.config.project_testing:
                print(f"Resampling features: {feature[:4]}...")

            original_length = feature.shape[0]
            original_indices = np.linspace(0, 1, original_length)
            target_indices = np.linspace(0, 1, target_length)

            # Interpolate each mel band independently
            interpolated_features = np.zeros((target_length, feature.shape[1]))

            for band in range(feature.shape[1]):
                interpolated_features[:, band] = np.interp(target_indices, original_indices, feature[:, band])

            resampled_features.append(interpolated_features)

        # Element-wise average across all resampled recordings
        keyword_template = np.mean(resampled_features, axis=0)

        if self.config.project_testing:
                print("Finished resampling features, template:" f"{keyword_template[:4]}...")

        return keyword_template

    def _characterize_keyword_audio(self):
        extracted_features = []
        for recording in self.filtered_keyword_recordings:
            extracted_features.append(self.features.extract_features(recording))

        self.keyword_template = self._build_keyword_template(extracted_features)

    def _isolate_speech_segment(self, filtered_audio_data):
        self.vad.reset_state()
        speech_start = None
        speech_end = None

        for start in range(0, len(filtered_audio_data) - self.config.frame_length, self.config.hop_length):
            frame = filtered_audio_data[start : start + self.config.frame_length]
            state = self.vad.detect_voice_activity(frame)

            if state == self.vad.VADState.SPEECH and speech_start is None:
                speech_start = start
            if state == self.vad.VADState.SILENCE and speech_start is not None:
                speech_end = start + self.config.frame_length
                break

        self.vad.reset_state()

        if speech_start is None:
            return None

        if speech_end is None:
            speech_end = len(filtered_audio_data)

        return filtered_audio_data[speech_start:speech_end]
