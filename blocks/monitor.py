#==================================================================================================#
# @name monitor.py
#
# @brief    Live monitor audio data, detect keyword, and trigger an event on keyword detection
#==================================================================================================#

import time

from audio.capture import AudioStream
from audio.filter import AudioFilter
from audio.capture import AudioRingBuffer
from audio.capture import KeywordAudioSetup
from processing.features import Features
from config import Config

from blocks.calibration import Calibration
from processing.vad import VoiceActivityDetection

class Monitor:
    def __init__(self, vad, keyword_template = None):
        self.audio_stream = AudioStream()
        self.audio_filter = AudioFilter()
        self.ring_buffer = AudioRingBuffer(int(16000 * 3)) # 3 second buffer capacity at 16kHz
        self.audio_detection_stream = KeywordAudioSetup()
        self.features = Features()
        self.vad = vad

        self.keyword_template = keyword_template

        # We take 25ms frames with 10ms hop, which is standard for speech processing tasks
        self.frame_length = 400
        self.hop_length = 160
        self.hangovers = 0
        self.keyword_frame_length = 0

    def _detected_keyword(self):
        # Placeholder for future implementation of keyword detection logic
        pass

    def _vad(self):
        # Placeholder for future implementation of voice activity detection logic
        pass

    def _process_speech_segment(self, segment):
        # Placeholder for future implementation of processing speech segment and checking for keyword
        pass

    def start_monitor(self):
        self.audio_stream.start_stream()

        # Wait for ring buffer to accumulate enough audio for filtering
        filter_context = 16000
        time.sleep(filter_context / 16000)

        self.keyword_frame_length = self.features.get_keyword_frame_length()
        print(f"Keyword frame length: {self.keyword_frame_length}")

        try:
            while True:
                time.sleep(self.hop_length / 16000)

                # Read a full second of audio — filter needs context to work
                audio_chunk = self.audio_stream.ring.read_audio(filter_context)

                # Filter the whole chunk, then take the last frame
                filtered = self.audio_filter.dc_offset_filter(audio_chunk)
                filtered = self.audio_filter.butterworth_bandpass_filter(filtered)
                filtered = self.audio_filter.preemphasize_filter(filtered)

                # Extract the last 400 samples (25 ms frame) for VAD
                frame = filtered[-self.frame_length:]
                state = self.vad.detect_voice_activity(frame)

                speech_duration = self.vad.get_vad_speech_duration()
                # print(f"speech duration: {speech_duration}")

                if speech_duration > self.keyword_frame_length:
                    print(f"Speech state detected, for {speech_duration * self.hop_length / 16000:.2f} seconds")

        except KeyboardInterrupt:
            print("Stopping monitor.")
        finally:
            self.audio_stream.stop_stream()


