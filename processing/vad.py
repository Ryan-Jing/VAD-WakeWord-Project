#==================================================================================================#
# @name vad.py
#
# @brief    Voice Activity Detection implementation for detecting speech segments in audio data
#==================================================================================================#

import numpy as np
from enum import Enum
from config import Config

class VADState(Enum):
    SILENCE = 0
    SPEECH = 1
    HANGOVER = 2

class VoiceActivityDetection:
    def __init__(self):
        self.config = Config()

        self.sampling_rate_hz = self.config.sampling_rate_hz
        self.frame_length = self.config.frame_length
        self.hop_length = self.config.hop_length
        self.smooth_window_length = 5

        self.VADState = VADState.SILENCE

        self.energy_top_threshold = None
        self.energy_bottom_threshold = None
        self.spectral_entropy_threshold = None
        self.hangovers_max = 4
        self.hangovers = 0

        self.start_frame = None
        self.current_frame = 0

        self.data_history = []
        self.energy_history = []
        self.entropy_history = []
        self.vad_speech_history_counter = 0
        self.previous_vad_state = self.VADState.SILENCE

    def reset_state(self):
        self.VADState = VADState.SILENCE
        self.hangovers = 0
        self.start_frame = None
        self.current_frame = 0

        self.data_history.clear()
        self.energy_history.clear()
        self.entropy_history.clear()
        self.vad_speech_history_counter = 0
        self.previous_vad_state = VADState.SILENCE

    def _get_energy(self, audio_data):
       return np.mean(audio_data ** 2)

    def _get_spectral_entropy(self, audio_data):
        # Shannon entropy of the normalized power spectrum
        windowed_data = audio_data * np.hamming(len(audio_data))
        spectrum = np.abs(np.fft.rfft(windowed_data)) ** 2
        spectrum = spectrum + 1e-12
        normalized_spectrum = spectrum / np.sum(spectrum)
        return -np.sum(normalized_spectrum * np.log2(normalized_spectrum))

    def _smooth_audio_data(self, audio_data, history):
        history.append(audio_data)
        if len(history) > self.smooth_window_length:
            history.pop(0)
        return np.mean(history)

    def _update_state(self, audio_data, energy, entropy):
        low_threhold_passed = energy > self.energy_bottom_threshold
        speech_detected = (energy > self.energy_top_threshold) and (entropy < self.spectral_entropy_threshold)

        if self.VADState == VADState.SILENCE:
            if speech_detected:
                self.VADState = VADState.SPEECH
                self.start_frame = self.current_frame

        elif self.VADState == VADState.SPEECH:
            if not low_threhold_passed:
                self.VADState = VADState.HANGOVER
                self.hangovers = 0

        elif self.VADState == VADState.HANGOVER:
            if low_threhold_passed:
                self.hangovers = 0
                self.VADState = VADState.SPEECH

            else:
                self.hangovers += 1
                if self.hangovers > self.hangovers_max:
                    self.VADState = VADState.SILENCE

        if self.VADState in (VADState.SPEECH, VADState.HANGOVER):
            self.vad_speech_history_counter += 1
        else:
            self.vad_speech_history_counter = 0

        self.previous_vad_state = self.VADState
        return self.VADState

    def vad_calibration(self, silence_audio_data):
        silence_audio_energy = []
        silence_audio_entropy = []

        for frame in range(0, len(silence_audio_data) - self.frame_length, self.hop_length):
            audio_frame = silence_audio_data[frame : frame + self.frame_length]
            silence_audio_energy.append(self._get_energy(audio_frame))
            silence_audio_entropy.append(self._get_spectral_entropy(audio_frame))

        average_energy = np.mean(silence_audio_energy)
        average_entropy = np.mean(silence_audio_entropy)

        self.energy_top_threshold = average_energy * 50
        self.energy_bottom_threshold = average_energy * 10
        self.spectral_entropy_threshold = average_entropy

    def detect_voice_activity(self, audio_data):
        frame_energy = self._get_energy(audio_data)
        frame_entropy = self._get_spectral_entropy(audio_data)

        smoothed_frame_energy = self._smooth_audio_data(frame_energy, self.energy_history)
        smoothed_frame_entropy = self._smooth_audio_data(frame_entropy, self.entropy_history)

        vad_state = self._update_state(audio_data, smoothed_frame_energy, smoothed_frame_entropy)

        self.current_frame += 1

        return vad_state

    def get_vad_speech_duration(self):
        return self.vad_speech_history_counter
