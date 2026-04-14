#==================================================================================================#
# @name utils.py
#
# @brief Utility functions for the project, for testing and debugging
#==================================================================================================#

import numpy as np
import matplotlib.pyplot as plt

class Utils:
    def plot_audio_data(self, audio, title="Audio Recording"):
        plt.figure(figsize=(10, 4))
        plt.plot(audio)
        plt.title(title)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def _plot_matrix(self, matrix, title, xlabel, ylabel, colorbar_label, cmap):
        plt.figure(figsize=(10, 6))

        matrix_to_plot = np.array(matrix, dtype=np.float64)
        matrix_to_plot[~np.isfinite(matrix_to_plot)] = np.nan
        masked_matrix = np.ma.masked_invalid(matrix_to_plot)

        image = plt.imshow(masked_matrix, aspect='auto', origin='lower', cmap=cmap)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        colorbar = plt.colorbar(image)
        colorbar.set_label(colorbar_label)
        plt.tight_layout()
        plt.show()

    def plot_frame_distance_matrix(
        self,
        frame_distance_matrix,
        title = "Frame-to-Frame Feature Distance",
        xlabel = "Live Utterance Frame",
        ylabel = "Keyword Template Frame",
        colorbar_label = "Euclidean Distance",
        cmap = "magma"
    ):
        self._plot_matrix(frame_distance_matrix, title, xlabel, ylabel, colorbar_label, cmap)

    # def plot_accumulated_cost_matrix(
    #     self,
    #     accumulated_cost_matrix,
    #     title = "DTW Accumulated Cost Matrix",
    #     xlabel = "Live Utterance Frame",
    #     ylabel = "Keyword Template Frame",
    #     colorbar_label = "Accumulated Cost",
    #     cmap = "viridis"
    # ):
    #     self._plot_matrix(accumulated_cost_matrix, title, xlabel, ylabel, colorbar_label, cmap)

    def plot_warping_path(
        self,
        accumulated_cost_matrix,
        warping_path,
        title = "DTW Warping Path",
        xlabel = "Live Utterance Frame",
        ylabel = "Keyword Template Frame",
        colorbar_label = "Accumulated Cost",
        cmap = "viridis",
        path_color = "red"
    ):
        plt.figure(figsize=(10, 6))

        matrix_to_plot = np.array(accumulated_cost_matrix, dtype=np.float64)
        matrix_to_plot[~np.isfinite(matrix_to_plot)] = np.nan
        masked_matrix = np.ma.masked_invalid(matrix_to_plot)

        image = plt.imshow(masked_matrix, aspect='auto', origin='lower', cmap=cmap)

        if len(warping_path) > 0:
            plt.plot(warping_path[:, 1], warping_path[:, 0], color=path_color, linewidth=2)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        colorbar = plt.colorbar(image)
        colorbar.set_label(colorbar_label)
        plt.tight_layout()
        plt.show()
