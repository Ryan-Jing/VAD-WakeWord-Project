#==================================================================================================#
# main.py
#
# Main file for project. Clean code will be my goal
#==================================================================================================#

import time

from audio.capture import AudioStream
from audio.filter import AudioFilter

from blocks.calibration import Calibration
from blocks.monitor import Monitor

def main():

    calibration = Calibration()

    print("Starting 544 Project")
    print("Starting calibration sequence")

    keyword_audio_recordings = calibration.calibration_sequence()

    time.sleep(1)

    print("Finished calibration sequence, starting monitor")
    monitor = Monitor(vad=calibration.vad)
    monitor.start_monitor()

if __name__ == "__main__":
    main()

