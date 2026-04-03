#==================================================================================================#
# @name capture.py
#
# @brief    Helper functions to get audio from the microphone
#           We want to capture audio, and create a ring buffer of 3 sceonds of audio
#==================================================================================================#

import numpy as np
import sounddevice as sd
import time

class AudioRingBuffer:
    def __init__(self, capacity):
        self._buffer = np.zeros(capacity, dtype = np.float32)
        self._capacity = capacity
        self._write_ptr = 0
        self._total_written_data = 0

    def write_audio(self, data_chunk):
        chunk_length = len(data_chunk)
        end_index = self._write_ptr + chunk_length

        if end_index <= self._capacity:
            self._buffer[self._write_ptr : end_index] = data_chunk
        else:
            first_index = self._capacity - self._write_ptr
            self._buffer[self._write_ptr :] = data_chunk[:first_index]
            self._buffer[:chunk_length - first_index] = data_chunk[first_index:]

        self._write_ptr = end_index % self._capacity
        self._total_written_data += chunk_length

    def read_audio(self, audio_chunk):
        chunk_length = min(audio_chunk, self._capacity, self._total_written_data)
        start_index = (self._write_ptr - audio_chunk) % self._capacity

        if start_index + chunk_length <= self._capacity:
            return self._buffer[start_index : start_index + chunk_length].copy()
        else:
            return np.concatenate([self._buffer[start_index :], self._buffer[: chunk_length - (self._capacity - start_index)]])

    def clear_audio(self):
        self._buffer[:] = 0.0
        self._write_ptr = 0
        self._total_written_data = 0

class AudioStream:
    def __init__(self, buffer_length_secs = 3):
        capacity = int(16000 * buffer_length_secs) # 16kHz audio, 3 second samples
        self.ring = AudioRingBuffer(capacity)
        self._audio_stream = sd.InputStream(
            samplerate = 16000,
            channels = 1,
            dtype = "float32",
            callback = self._callback
            )

    def _callback(self, input_data, num_frames, time, status):
        self.ring.write_audio(input_data[:, 0])

    def start_stream(self):
        self._audio_stream.start()

    def stop_stream(self):
        self._audio_stream.stop()

class KeywordAudioSetup:
    def __init__(self, buffer_length_secs = 3):
        self.audio_stream = AudioStream(buffer_length_secs)

    def get_audio_chunk(self, chunk_length_secs = 1):
        chunk_length = int(16000 * chunk_length_secs)
        return self.audio_stream.ring.read_audio(chunk_length)

    def record_utterance(self, record_duration = 3):
        self.audio_stream.ring.clear_audio()
        self.audio_stream.start_stream()
        time.sleep(record_duration)

        recorded_audio = self.audio_stream.ring.read_audio(int(16000 * record_duration))
        self.audio_stream.stop_stream()
        return recorded_audio

    def record_silence(self, record_duration = 3):
        return self.record_utterance(record_duration)
