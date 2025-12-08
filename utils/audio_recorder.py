import base64
import io
import typing as tp
import uuid
from concurrent.futures import Future
from pathlib import Path

import ffmpeg
import ipywidgets as widgets
import jinja2
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from IPython.display import clear_output, display, HTML


_env = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))


class AudioRecorder:
    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_duration_ms: int = 30,
        record_timeout_sec: float = 5,
        output_widget: tp.Optional[widgets.Output] = None,
    ) -> None:
        self.template = _env.get_template("audio_recorder.html.j2")
        self.sampling_rate = sampling_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.sampling_rate * (self.frame_duration_ms / 1000.0))
        self.bytes_per_sample = 2
        self.bytes_per_frame = self.frame_size * self.bytes_per_sample
        self.record_timeout_sec = record_timeout_sec

        self.unique_id = f"audio-receiver-{str(uuid.uuid4())}"
        self.receiver_css_class = f"audio-data-receiver-{self.unique_id}"

        self.recording_result = None

        self.data_receiver = widgets.Text(
            value="",
            description="",
            disabled=False,
            layout=widgets.Layout(display="none"),
            style={"description_width": "0px"},
        )
        self.data_receiver.add_class(self.receiver_css_class)

        self.output_widget = output_widget
        self.data_receiver.observe(self._handle_received_data, names="value")
        self.result_future = Future()

    def record(self) -> Future:
        output_widget = self.output_widget if self.output_widget else widgets.Output()
        with output_widget:
            output_widget.clear_output(wait=True)
            rendered_html = self.template.render(
                unique_id=self.unique_id, css_class_receiver=self.receiver_css_class
            )
            display(HTML(rendered_html), self.data_receiver)

        if self.output_widget is None:
            display(output_widget)

        return self.result_future

    def _handle_received_data(self, change: dict[str, tp.Any]) -> None:
        try:
            base64_data_url = change["new"]
            if not base64_data_url or not base64_data_url.startswith("data:audio"):
                raise ValueError(f"Got response: {base64_data_url}")

            header, encoded = base64_data_url.split(",", 1)
            audio_binary = base64.b64decode(encoded)

            mime_type = header.split(":")[1].split(";")[0]
            input_format = mime_type.split("/")[1]

            process = (
                ffmpeg.input("pipe:0", f=input_format)
                .output(
                    "pipe:1",
                    format="wav",
                    acodec="pcm_s16le",
                    ac=1,
                    ar=self.sampling_rate,
                )
                .run_async(
                    pipe_stdin=True,
                    pipe_stdout=True,
                    pipe_stderr=True,
                    overwrite_output=True,
                )
            )
            output_bytes, err_bytes = process.communicate(input=audio_binary)
            if err_bytes:
                # TODO: log stderr here
                # raise RuntimeError(f"FFmpeg warning/error: {err_bytes.decode()}")
                pass
            if not output_bytes:
                raise ValueError("FFmpeg produced no output. Conversion failed.")

            wav_data = bytearray(output_bytes)
            riff_size = len(wav_data) - 8
            wav_data[4:8] = riff_size.to_bytes(4, byteorder="little")

            original_sr, audio_data = wavfile.read(io.BytesIO(wav_data))

            if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                axis_to_average = 1 if audio_data.shape[1] < audio_data.shape[0] else 0
                audio_data = audio_data.mean(axis=axis_to_average).astype(
                    audio_data.dtype
                )

            if original_sr != self.sampling_rate:
                audio_data = resample_poly(audio_data, self.sampling_rate, original_sr)

            if audio_data.dtype != np.int16:
                if np.issubdtype(audio_data.dtype, np.floating):
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val
                    audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(
                        np.int16
                    )
                elif np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)

            self.result_future.set_result(audio_data)
        except Exception as e:
            self.result_future.set_exception(e)
