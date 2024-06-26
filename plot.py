import queue
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sounddevice as sd
from sounddevice import CallbackFlags
import sys

# Constants
DEVICE_NO = 1
WINDOW_MS = 500
UPDATE_INTERVAL_MS = 20
MAPPING = [0]  # mono
DOWNSAMPLE = 1
CHANNELS = [1]
TARGET_SAMPLERATE = 16000  # Set target samplerate to 16000

class AudioPlotter:
    def __init__(self):
        self.q = queue.Queue()
        self.chunk_len = self._calculate_length()
        print(f"Chunk length: {self.chunk_len}, (s) {self.chunk_len / TARGET_SAMPLERATE}")

        self.plotdata = np.zeros((self.chunk_len, len(CHANNELS)))
        self.lines = None

    def _calculate_length(self) -> int:
        return int(WINDOW_MS * TARGET_SAMPLERATE / (1000 * DOWNSAMPLE))

    def audio_callback(self, indata: np.ndarray, frames: int, time, status: CallbackFlags):
        if status: print(status, file=sys.stderr)

        # texts = transcribe(indata)
        # print(texts)

        self.q.put(indata[::DOWNSAMPLE, MAPPING])

    def update_plot(self, frame: int):
        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break

            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines

    def _init_plot(self) -> plt.Figure:
        fig, ax = plt.subplots()
        self.lines = ax.plot(self.plotdata)
        if len(CHANNELS) > 1:
            ax.legend([f'channel {c}' for c in CHANNELS],
                      loc='lower left', ncol=len(CHANNELS))
        ax.axis((0, len(self.plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)
        return fig

    def prepare_plot(self):
        fig = self._init_plot()
        return FuncAnimation(fig, self.update_plot, interval=UPDATE_INTERVAL_MS, blit=True, cache_frame_data=False)

    def plot(self):
        try:
            fig = self._init_plot()
            ani = FuncAnimation(fig, self.update_plot, interval=UPDATE_INTERVAL_MS, blit=True, cache_frame_data=False)

            stream = sd.InputStream(
                device=DEVICE_NO, channels=max(CHANNELS),
                samplerate=TARGET_SAMPLERATE, callback=self.audio_callback, dtype=np.float32)

            with stream:
                plt.show()

        except Exception as e:
            print(f"Error: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    audio_plotter = AudioPlotter()
    audio_plotter.plot()
