#==================================================================================================#
# @name main.py
#
# @brief Main file for project. Clean code will be my goal
#==================================================================================================#

import time
import matplotlib.pyplot as plt

from blocks.calibration import Calibration
from blocks.monitor import Monitor

def plot_keyword_template(keyword_template):
    plt.figure(figsize=(10, 4))
    plt.imshow(keyword_template.T, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Frame (time)')
    plt.ylabel('Mel Band')
    plt.title('Averaged Keyword Template')
    plt.colorbar(label='Log Energy')
    plt.tight_layout()
    plt.show()

def main():

    calibration = Calibration()

    print("Starting 544 Project")
    print("Starting calibration sequence")

    calibration.calibration_sequence()
    print("Finished calibration sequence")

    keyword_template = calibration.get_keyword_template()
    dtw_distance_threshold = calibration.get_dtw_distance_threshold()
    print("Keyword template shape:", keyword_template.shape) #type: ignore
    print(f"DTW threshold: {dtw_distance_threshold:.4f}")
    print("Received keyword template, plotting...")

    plot_keyword_template(keyword_template)

    time.sleep(1)

    print("=" * 50)
    print("Finished calibration sequence, starting monitor")
    monitor = Monitor(calibration.vad, keyword_template, features=calibration.features, dtw_distance_threshold = dtw_distance_threshold)
    monitor.start_monitor()

    return

if __name__ == "__main__":
    main()
