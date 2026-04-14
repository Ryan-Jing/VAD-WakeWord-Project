#==================================================================================================#
# @name monitor.py
#
# @brief    Live monitor audio data, detect keyword, and trigger an event on keyword detection
#==================================================================================================#

import time

from audio.capture import AudioStream
from audio.capture import AudioRingBuffer
from audio.filter import AudioFilter

from processing.vad import VADState
from processing.dtw import DTW

from config import Config
from utils import Utils

class Monitor:
    def __init__(self, vad, keyword_template, features, dtw_distance_threshold = None):
        self.config = Config()
        self.audio_stream = AudioStream()
        self.audio_filter = AudioFilter()
        self.ring_buffer = AudioRingBuffer(int(self.config.sampling_rate_hz * 3))
        self.features = features
        self.dtw = DTW(keyword_template, distance_threshold = dtw_distance_threshold)
        self.vad = vad
        self.utils = Utils()

        self.keyword_template = keyword_template

        # We take 25ms frames with 10ms hop, which is standard for speech processing tasks
        self.frame_length = self.config.frame_length
        self.hop_length = self.config.hop_length
        self.frame_buffer = AudioRingBuffer(self.frame_length)
        self.hangovers = 0
        self.keyword_frame_length = 0

    def start_monitor(self):
        previous_state = None
        previous_speech_duration = 0

        self.audio_stream.ring.clear_audio()
        self.ring_buffer.clear_audio()
        self.frame_buffer.clear_audio()
        self.audio_filter.reset_streaming_state()
        self.vad.reset_state()

        self.audio_stream.start_stream()
        read_cursor = self.audio_stream.ring.get_total_written_data()
        self.keyword_frame_length = self.features.get_keyword_frame_length()
        print(f"Keyword frame length: {self.keyword_frame_length}")
        print(f"DTW threshold: {self.dtw.distance_threshold:.4f}")

        try:
            while True:
                time.sleep(self.hop_length / self.config.sampling_rate_hz)
                total_written = self.audio_stream.ring.get_total_written_data()

                while total_written - read_cursor >= self.hop_length:
                    audio_chunk = self.audio_stream.ring.read_audio_from(read_cursor, self.hop_length)
                    if len(audio_chunk) == 0:
                        read_cursor = total_written
                        break

                    read_cursor += len(audio_chunk)

                    filtered_chunk = self.audio_filter.filter_audio_chunk(audio_chunk)
                    self.ring_buffer.write_audio(filtered_chunk)
                    self.frame_buffer.write_audio(filtered_chunk)

                    if self.frame_buffer.get_total_written_data() < self.frame_length:
                        total_written = self.audio_stream.ring.get_total_written_data()
                        continue

                    frame = self.frame_buffer.read_audio(self.frame_length)
                    state = self.vad.detect_voice_activity(frame)

                    speech_duration = self.vad.get_vad_speech_duration()
                    if speech_duration > 0:
                        print(f"VAD state: {state}, speech duration: {speech_duration}")

                    if previous_state == VADState.HANGOVER and state == VADState.SILENCE:
                        if previous_speech_duration > self.keyword_frame_length * 0.5:
                            speech_samples = self.frame_length + max(previous_speech_duration - 1, 0) * self.hop_length
                            print(f"Speech ended, duration: {speech_samples / self.config.sampling_rate_hz:.2f}s")

                            speech_filtered = self.ring_buffer.read_audio(speech_samples)
                            extracted_features = self.features.extract_features(speech_filtered)
                            detected, dtw_distance, dtw_analysis = self.dtw.is_keyword_detected(extracted_features, return_analysis = True) #type: ignore
                            print(f"DTW distance: {dtw_distance:.4f}")

                            if self.config.project_testing:
                                detection_status = "Detected Keyword" if detected else "Rejected Utterance"
                                dtw_title_prefix = f"{detection_status} | DTW Distance: {dtw_distance:.4f}"

                                self.utils.plot_frame_distance_matrix(
                                    dtw_analysis["frame_distance_matrix"],
                                    title = f"{dtw_title_prefix} | Frame-to-Frame Comparison",
                                    xlabel = "Live Utterance Frame Index",
                                    ylabel = "Keyword Template Frame Index"
                                )
                                # self.utils.plot_accumulated_cost_matrix(
                                #     dtw_analysis["accumulated_cost_matrix"],
                                #     title = f"{dtw_title_prefix} | Accumulated Cost Matrix",
                                #     xlabel = "Live Utterance Frame Index",
                                #     ylabel = "Keyword Template Frame Index"
                                # )
                                self.utils.plot_warping_path(
                                    dtw_analysis["accumulated_cost_matrix"],
                                    dtw_analysis["warping_path"],
                                    title = f"{dtw_title_prefix} | Warping Path",
                                    xlabel = "Live Utterance Frame Index",
                                    ylabel = "Keyword Template Frame Index"
                                )

                            if detected:
                                print(f"Live features shape: {extracted_features.shape}")
                                print("*** KEYWORD DETECTED ***")

                    previous_speech_duration = speech_duration
                    previous_state = state
                    total_written = self.audio_stream.ring.get_total_written_data()

        except KeyboardInterrupt:
            print("Stopping monitor.")
        finally:
            self.audio_stream.stop_stream()
