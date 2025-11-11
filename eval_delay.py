import numpy as np
import torch
import torchaudio
import transformers

from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, load_voices

from typing import Sequence, Tuple, Union

import re
import time
import latex2mathml.converter
import subprocess

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


class TTSEvaluator:
    def __init__(self):
        self.tts = TextToSpeech(kv_cache=True, use_deepspeed=True, half=True)

    def _inference(
        self,
        text,
        script,
        voice,
        voice_b,
        seed,
        split_by_newline,
    ):
        if text is None or text.strip() == "":
            with open(script.name) as f:
                text = f.read()
            if text.strip() == "":
                raise gr.Error("Please provide either text or script file with content.")

        if split_by_newline == "Yes":
            texts = list(filter(lambda x: x.strip() != "", text.split("\n")))
        else:
            texts = split_and_recombine_text(text)

        voices = [voice]
        if voice_b != "disabled":
            voices.append(voice_b)

        if len(voices) == 1:
            voice_samples, conditioning_latents = load_voice(voice)
        else:
            voice_samples, conditioning_latents = load_voices(voices)

        for j, text in enumerate(texts):
            total_audio_frame = []
            for audio_frame in self.tts.tts_with_preset(
                text,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset="ultra_fast",
                k=1,
                verbose=False
            ):
                total_audio_frame.append(audio_frame.cpu().detach().numpy())
            yield (24000, np.concatenate(total_audio_frame, axis=0))
    
    @staticmethod
    def clearspeak(mathml: str) -> str:
        result = subprocess.run(
            ["./node_modules/.bin/sre"],
            input=mathml,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout

    @staticmethod
    def convert_markdown_with_latex(text: str) -> str:
        def replace_math(match):
            latex = (match.group(1) or match.group(2)).strip()
            mathml = latex2mathml.converter.convert(latex)
            return TTSEvaluator.clearspeak(mathml)[:-1]

        pattern = re.compile(r"\$\$([^$]+)\$\$|\$([^$]+)\$")
        return re.sub(pattern, replace_math, text)
    
    @staticmethod
    def chunk_tokens_with_latex(token_times, k=5):
        chunks = []
        current_tokens, current_times = [], []
        inside_math = False
        delimiter = None  # "$" or "$$"

        def flush():
            if not current_tokens:
                return
            chunks.append({
                'text': ''.join(current_tokens).strip().replace('\n', ""),
                'times': current_times[:]
            })
            current_tokens.clear()
            current_times.clear()

        i = 0
        total_tokens = len(token_times)
        def next_is_dollar(i):
            return (i + 1 < total_tokens) and (token_times[i + 1][0].strip() == "$")

        while i < total_tokens:
            tok, t = token_times[i]
            # We live in assumption that if tok contains $ then tok.strip() either "$" or "$$"
            stripped = tok.strip()
            
            current_tokens.append(tok)
            current_times.append(t)

            if not inside_math:
                if stripped == "$$":
                    inside_math, delimiter = True, "$$"
                elif stripped == "$":
                    if next_is_dollar(i):
                        # "$" + "$" => "$$" opener
                        inside_math, delimiter = True, "$$"
                        i += 1  # consume the second "$"
                        # also record it
                        tok2, t2 = token_times[i]
                        current_tokens.append(tok2)
                        current_times.append(t2)
                    else:
                        inside_math, delimiter = True, "$"
            else:
                if delimiter == "$$":
                    if stripped == "$$":
                        inside_math, delimiter = False, None
                    elif stripped == "$" and next_is_dollar(i):
                        # "$" + "$" => "$$" closer
                        inside_math, delimiter = False, None
                        i += 1  # consume the second "$"
                        tok2, t2 = token_times[i]
                        current_tokens.append(tok2)
                        current_times.append(t2)
                else:  # delimiter == "$"
                    if stripped == "$":
                        inside_math, delimiter = False, None
            i += 1

            if len(current_tokens) >= k and not inside_math:
                flush()

        flush()
        return chunks
    
    @staticmethod
    def compute_delays(chunk_done_relative_timestamps: Sequence[float],
                        chunk_audio_durations: Sequence[float]) -> float:
        """
        :param chunk_done_relative_timestamps: for each generated speech chunk,
            this is how long (time) passed between user request and when this chunk
            was ready to be voiced (including llm, tts, etc).
        :param chunk_audio_durations: the audio length of this individual chunk
            (not cumulative, not accounting for LLM / TTS - just the audio alone)
        :returns: user-perceived delay before each chunk

        :example:
        >>> compute_delay([1, 3, 8], [5, 1, 3])  # [1.0, 0, 1.0]
        """
        assert len(chunk_done_relative_timestamps) == len(chunk_audio_durations)
        delays = []
        earliest_next_chunk_start = 0.0
        for chunk_done_by, chunk_audio_duration in zip(
            chunk_done_relative_timestamps, chunk_audio_durations):
            real_chunk_start = max(earliest_next_chunk_start, chunk_done_by)
            # ^-- when the user actually starts hearing this audio, with all delays
            delays.append(real_chunk_start - earliest_next_chunk_start)
            earliest_next_chunk_start = real_chunk_start + chunk_audio_duration
        return delays
    
    def get_audio_track(self, text):
        # !!! Sometimes this code fails due to matrix dim mismatch (that is something wrong with tortois-tts). Just rerun cell.
        # Here is temporary solution
        flag = True
        while flag:
            try:
                frames_srate = []
                spk_times = []
                tts_times = []
                t0 = time.perf_counter()
                for sample_rate, frame in self._inference(
                    text=text,
                    script=None,
                    voice="freeman",
                    voice_b="disabled",
                    seed=42,
                    split_by_newline="Yes",
                ):
                    t1 = time.perf_counter()
                    spk_times.append(len(frame) / sample_rate)
                    tts_times.append(t1 - t0)
                    frames_srate.append((frame, sample_rate))
                    t0 = time.perf_counter()
                flag = False
            except RuntimeError as RE:
                logger.debug(f"Caught RuntimeError: {RE}")

        total_frame = np.concatenate([el[0] for el in frames_srate], axis=0)
        return total_frame, frames_srate[0][1], spk_times, tts_times

    def __call__(self,
        token_times: Sequence[Tuple[str, float]],
        k_chunks: int = 5,
        add_tts_in_parrallel: bool = True,
        return_chunks: bool = False,
        return_audio: bool = False,
        mock_spk_times: Union[None, Sequence[float]] = None,
        mock_tts_times: Union[None, Sequence[float]] = None,
    ):
        """
        Here will be better doc string
        token_times: Sequence[(decoded_str, generated_timestamp)]
                        ^-- eos included
        """ 

        # Chunking with latex
        chunked_token_times = self.chunk_tokens_with_latex(token_times[:-1], k=k_chunks)
        texts = [el["text"] for el in chunked_token_times]
        gen_times = [el["times"][-1] for el in chunked_token_times]
        chunk_sizes = [len(el["times"]) for el in chunked_token_times]

        # Calling tts or use mock values
        assert not ((mock_spk_times is None) ^ (mock_tts_times is None)), "You must use either both or neither: mock_spk_times, mock_tts_times"
        if mock_spk_times is None:
            text = self.convert_markdown_with_latex("\n".join(texts))
            total_frame, frame_rate, spk_times, tts_times = self.get_audio_track(text)
        else:
            assert not return_audio, "Cannot return audio with mock tts times"
            spk_times, tts_times = mock_spk_times, mock_tts_times

        assert len(gen_times) == len(spk_times), f"{len(gen_times)}, {len(spk_times)}"
        assert len(gen_times) == len(tts_times), f"{len(gen_times)}, {len(tts_times)}"

        if add_tts_in_parrallel:
            chunk_ready = np.array(gen_times) + np.array(tts_times)
        else:
            chunk_ready = np.array(gen_times) + np.cumsum(tts_times)
        
        delays = self.compute_delays(chunk_ready, spk_times)

        metrics = {
            "total_delay": float(np.sum(delays)),
            "delays": delays,
            "duration_no_delay": float(np.sum(spk_times)),
            "duration_with_delay": float(np.sum(spk_times) + float(np.sum(delays))),
        }
        output = [metrics]
        if return_chunks:
            output.append({
                "text_chunks": texts,
                "chunk_sizes": chunk_sizes,
                "gen_timestamps_chunks": gen_times,
                "tts_times_chunks": tts_times,
                "spk_times_chunks": spk_times, 
                }
            )
        if return_audio:
            output.append({
                "frame": total_frame,
                "frame_rate": frame_rate,
                }
            )
        return output
