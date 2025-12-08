import logging
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import ipywidgets as widgets
import numpy as np
import torch
import transformers
import whisper
from IPython.display import clear_output, display, Audio

from async_reasoning.cache import State
from async_reasoning.solver import AsyncReasoningSolver as Solver
from evals.tts_evaluator import TTSEvaluator
from utils.audio_recorder import AudioRecorder

logger = logging.getLogger(__name__)
logging.basicConfig(filename="demo.log", encoding="utf-8", level=logging.DEBUG)


class VoiceAssistant:
    def __init__(
        self,
        asr_model_name: str = "base",
        llm_model_name: str = "Qwen/Qwen3-14B",
        use_fast_kernel: bool = True,
        tokens_per_chunk: int = 5,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model_name)
        llm_model = transformers.AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=device,
        )
        # forbidden_token_ix are Model-dependent
        forbidden_token_ix = [
            tokenizer.vocab[x] for x in ("</think>", "<|im_start|>", "SYSTEM")
        ]
        self.solver = Solver(
            llm_model,
            tokenizer,
            forbidden_token_ix,
            use_fast_kernel=use_fast_kernel,
        )
        self.asr_model = whisper.load_model(asr_model_name)
        self.tts_evaluator = TTSEvaluator()

        self.record_button = widgets.Button(
            description="Record voice",
            button_style="info",
        )
        self.output = widgets.Output()
        self.queue = Queue()
        self.tokens_per_chunk = tokens_per_chunk

    def display(self) -> None:
        self.output.clear_output()
        clear_output(True)

        self.record_button.layout.display = None  # Make the button visible
        self.record_button.on_click(self._record)
        display(self.record_button, self.output)

    def _record(self, _) -> None:
        self.record_button.layout.display = "none"
        audio_data_future = AudioRecorder(output_widget=self.output).record()
        audio_data_future.add_done_callback(self._recording_done)

    def _recording_done(self, future) -> None:
        audio_data = future.result()
        audio_data = audio_data.astype(np.float32) / 32768.0

        with self.output:
            clear_output()
            print("Transcribing audio...")
            result = self.asr_model.transcribe(audio_data, language="en")
            print("Transcribed result: ", result["text"])

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._output_loop)
                self.solver.solve(
                    result["text"],
                    budget=1024,
                    display_generation_in_real_time=False,
                    on_new_tokens_generated=self._new_tokens_generated,
                )
                future.result()

    def _new_tokens_generated(
        self,
        writer_output_str: str,
        thinker_output_str: str,
        token_times: tuple[str, float, int],
        eos_generated: bool,
        state: State,
    ) -> None:
        self.queue.put(
            (writer_output_str, thinker_output_str, token_times, eos_generated, state)
        )

    def _output_loop(self) -> None:
        next_chunk_end_time = float("-inf")
        last_played_chunks = []
        while True:
            available = self.queue.qsize()
            for _ in range(max(available, 1)):
                (
                    writer_output_str,
                    thinker_output_str,
                    token_times,
                    eos_generated,
                    state,
                ) = self.queue.get()
            received_chunks = self.tts_evaluator.chunk_tokens_with_latex(
                token_times, k=self.tokens_per_chunk
            )
            if len(last_played_chunks) + 1 < len(received_chunks) or eos_generated:
                if not eos_generated:
                    new_chunks = received_chunks[len(last_played_chunks) : -1]
                else:
                    new_chunks = received_chunks[len(last_played_chunks) :]
                    if len(new_chunks) == 0:
                        break

                texts = [
                    self.tts_evaluator.convert_markdown_with_latex(el["text"])
                    for el in new_chunks
                ]
                logger.debug(f"Playing new chunk: {texts}")
                total_frame, frame_rate, _, _ = self.tts_evaluator.get_audio_track(
                    texts
                )
                audio = {
                    "frame": total_frame,
                    "frame_rate": frame_rate,
                }
                duration = total_frame.shape[0] / audio["frame_rate"]
                sleep_time = max(0, next_chunk_end_time - time.time())
                logger.debug(
                    f"New chunk duration: {duration}, sleeping for {sleep_time} seconds"
                )
                time.sleep(sleep_time)
                with self.output:
                    self.solver.display_tokens(
                        writer_output_str, thinker_output_str, state
                    )
                    last_played_chunks = last_played_chunks + new_chunks
                    audio = Audio(
                        audio["frame"], rate=audio["frame_rate"], autoplay=True
                    )
                    display(audio)
                next_chunk_end_time = time.time() + duration

            if eos_generated:
                break
