#==================================================================================================#
# @name calibration.py
#
# @brief    Calibration and training of keyword spotting pipeline
#==================================================================================================#

import time

from audio.capture import AudioStream
from audio.filter import AudioFilter
from audio.capture import KeywordAudioSetup
from processing.vad import VoiceActivityDetection

class Calibration:
    def __init__(self):
        self.audio_stream = AudioStream()
        self.audio_filter = AudioFilter()
        self.calibration = KeywordAudioSetup()
        self.vad = VoiceActivityDetection()

        self.keyword_audio_recordings = []
        self.filtered_keyword_recordings = []
        self.keyword_template = None

    def calibration_sequence(self, duration_secs = 3, num_keyword_recordings = 3):
        print("Recording silence for calibration, please be quiet")
        silence_audio = self.calibration.record_silence(duration_secs)
        silence_filtered = self.audio_filter.dc_offset_filter(silence_audio)
        silence_filtered = self.audio_filter.butterworth_bandpass_filter(silence_filtered)
        silence_filtered = self.audio_filter.preemphasize_filter(silence_filtered)
        self.vad.vad_calibration(silence_filtered)

        # for number in range(num_keyword_recordings):
        #     print(f"Recording keyword for calibration, please say the keyword ({number + 1}/{num_keyword_recordings})...")
        #     self.keyword_audio_recordings.append(self.calibration.record_utterance(duration_secs))

        #     print("Finished recording")
        #     time.sleep(1)

        return self.keyword_audio_recordings

    def filter_calibration_audio(self):
        for recording in self.keyword_audio_recordings:
            dc_filtered = self.audio_filter.dc_offset_filter(recording)
            bw_filtered = self.audio_filter.butterworth_bandpass_filter(dc_filtered)
            preemphasized_audio = self.audio_filter.preemphasize_filter(bw_filtered)

            self.filtered_keyword_recordings.append(preemphasized_audio)

    def characterize_keyword_audio(self):
        # Placeholder for future implementation of characterizing keyword audio
        pass

    def get_keyword_template(self):
        # Placeholder for future implementation of getting keyword template
        return self.keyword_template




