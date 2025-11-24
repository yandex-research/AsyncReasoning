import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
import transformers

from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, load_voices

from typing import Sequence, List, Dict, Any, Tuple, Union

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

    @staticmethod
    def clearspeak(mathml: str) -> str:
        """
        Convert a MathML expression into a human-readable spoken string using
        the Speech Rule Engine (SRE).

        :param mathml: MathML markup string representing a mathematical expression.
        :returns: The spoken (ClearSpeak) verbalization of the given MathML expression.

        :example:
        >>> clearspeak("<math><mfrac><mn>1</mn><mn>2</mn></mfrac></math>")
        'one half'
        """
        result = subprocess.run(
            [f"{Path(__file__).resolve().parent}/../node_modules/.bin/sre"],
            input=mathml,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout

    @staticmethod
    def convert_markdown_with_latex(text: str) -> str:
        """
        Convert all LaTeX formulas within a Markdown text into their spoken
        (ClearSpeak) equivalents by transforming them through MathML.

        This function finds both inline (`$...$`) and block (`$$...$$`) LaTeX
        expressions, converts them to MathML, then uses `TTSEvaluator.clearspeak`
        to produce human-readable speech text.

        :param text: Markdown string possibly containing LaTeX expressions.
        :returns: Markdown text with LaTeX replaced by corresponding spoken text.

        :example:
        >>> convert_markdown_with_latex("The result is $1 + 2 = 3$.")
        'The result is one plus two equals three.'
        """
        def replace_math(match):
            latex = (match.group(1) or match.group(2)).strip()
            mathml = latex2mathml.converter.convert(latex)
            return TTSEvaluator.clearspeak(mathml)[:-1]

        # Removed boxed voicing, it does not work well with clearspeak
        text = re.sub(r'\\boxed\s*{([^}]*)}', r'\1', text)
        pattern = re.compile(r"\$\$([^$]+)\$\$|\$([^$]+)\$")
        return re.sub(pattern, replace_math, text)
    
    @staticmethod
    def chunk_tokens_with_latex(token_times, k=5) -> List[Dict[str, Any]]:
        """
        Group tokens into chunks of size `k` while ensuring that entire LaTeX math
        expressions (delimited by `$...$` or `$$...$$`) remain in the same chunk.

        Each chunk contains:
        - `text`: concatenated string of tokens in that chunk
        - `times`: list of corresponding token generation timestamps

        :param token_times: List of (token, timestamp, step) triplets representing text tokens
            and the time and step when each was generated.
        :param k: Target number of tokens per chunk, excluding LaTeX grouping
            constraints.
        :returns: List of chunks, where each chunk is a dict with 'text', 'times', 'steps'.

        :example:
        >>> token_times = [
        ...     ("We", 0.1), (" ", 0.2), ("are", 0.3),
        ...     (" ", 0.4), ("$", 0.5), ("x", 0.6), ("+", 0.7),
        ...     ("y", 0.8), ("$", 0.9), (" ", 1.0), ("done", 1.1)
        ... ] # and also step indices: 0, 1, 2, 3, 4, 5 ...
        >>> chunk_tokens_with_latex(token_times, k=5)
        [{'text': 'We are $x+y$', 'times': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'steps': [0, 1, 2, 3, 4..]},
        {'text': 'done', 'times': [1.0, 1.1], 'steps': [10, 11, 12, 13, 14..]},
        ]
        """
        chunks = []
        current_tokens, current_times, current_steps = [], [], []
        inside_math = False
        delimiter = None  # "$" or "$$"

        def flush():
            if not current_tokens:
                return
            chunks.append({
                'text': ''.join(current_tokens).strip().replace('\n', ""),
                'times': current_times[:],
                'steps': current_steps[:],
            })
            current_tokens.clear()
            current_times.clear()
            current_steps.clear()

        i = 0
        total_tokens = len(token_times)
        def next_is_dollar(i):
            return (i + 1 < total_tokens) and (token_times[i + 1][0].strip().startswith("$"))

        while i < total_tokens:
            tok, t, s = token_times[i]
            stripped = tok.strip()
            assert (not "$" in stripped) or ("$$" in stripped and stripped.count("$") == 2) or ("$" in stripped and stripped.count("$") == 1), f"{stripped=}, {tok=}"
            
            current_tokens.append(tok)
            current_times.append(t)
            current_steps.append(s)

            if not inside_math:
                if "$$" in stripped:
                    inside_math, delimiter = True, "$$"
                elif "$" in stripped:
                    if next_is_dollar(i):
                        # "$" + "$" => "$$" opener
                        inside_math, delimiter = True, "$$"
                        i += 1  # consume the second "$"
                        # also record it
                        tok2, t2, s2 = token_times[i]
                        current_tokens.append(tok2)
                        current_times.append(t2)
                        current_steps.append(s2)
                    else:
                        inside_math, delimiter = True, "$"
            else:
                if delimiter.startswith("$$"):
                    if "$$" in stripped:
                        inside_math, delimiter = False, None
                    elif "$" in stripped and next_is_dollar(i):
                        # "$" + "$" => "$$" closer
                        inside_math, delimiter = False, None
                        i += 1  # consume the second "$"
                        tok2, t2, s2 = token_times[i]
                        current_tokens.append(tok2)
                        current_times.append(t2)
                        current_steps.append(s2)
                else:  # delimiter == "$"
                    if "$" in stripped:
                        inside_math, delimiter = False, None
            i += 1

            if len(current_tokens) >= k and not inside_math:
                flush()

        flush()
        return chunks
    
    @staticmethod
    def compute_delays(chunk_done_relative_timestamps: Sequence[float],
                        chunk_audio_durations: Sequence[float]) -> Sequence[float]:
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

    def _inference(
        self,
        text,
        voice,
        voice_b,
        seed,
        split_by_newline,
    ):
        """
        Slightly altered inference function from tortois-tts
        """
        assert text is not None and text.strip() != "", f"Provide text please."

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
                verbose=False,
                overlap_wav_len=1, # This is not the best way to do that!
            ):
                total_audio_frame.append(audio_frame.cpu().detach().numpy())
            yield (24000, np.concatenate(total_audio_frame, axis=0))

    def get_audio_track(self, 
        texts: Sequence[str],
        k_chunks: int,
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        Generate and concatenate audio tracks for a sequence of text chunks using a TTS model.

        The function performs iterative inference through `self._inference` for each text input.
        It accumulates generated audio frames, tracks per-chunk TTS generation time, and
        computes playback durations.  
        If a transient `RuntimeError` occurs (e.g., matrix dimension mismatch from Tortoise-TTS),
        the process is automatically retried until success.

        :param texts: Sequence of text chunks to synthesize speech for.
        :returns:
            - total_frame: Concatenated NumPy array of all generated audio frames.
            - sample_rate: Sampling rate of the generated audio.
            - spk_times: List of audio playback durations (in seconds) for each chunk.
            - tts_times: List of TTS generation durations (in seconds) for each chunk.

        :example:
        >>> total_frame, sr, spk_times, tts_times = tts_engine.get_audio_track(["Hello", "world"])
        >>> sr
        22050
        >>> len(spk_times)
        2
        """
        
        frames = []
        spk_times = []
        tts_times = []
        k_chunks = 3 * k_chunks # make formulas sound more natural
        for text in texts:
            if text is None or text.strip() == "":
                tts_times.append(0)
                spk_times.append(0)
                continue

            t0 = time.perf_counter()
            spk_time = 0
            
            words = text.split(" ")
            chunks = []
            for i in range(0, len(words), k_chunks):
                chunks.append(" ".join(words[i:i+k_chunks]))
            splitted_text = "\n".join(chunks)

            prev_sample_rate = None
            for sample_rate, frame in self._inference(
                text=splitted_text,
                voice="freeman",
                voice_b="disabled",
                seed=42,
                split_by_newline="Yes",
            ):
                if prev_sample_rate is not None:
                    assert sample_rate == prev_sample_rate, f"Got unexpected {sample_rate=}, expected the same from prev chunk {prev_sample_rate=}"
                prev_sample_rate = sample_rate
                
                spk_time += len(frame) / sample_rate
                frames.append(frame)

            t1 = time.perf_counter()
            tts_times.append(t1 - t0)
            spk_times.append(spk_time)
        total_frame = np.concatenate(frames, axis=0)
        return total_frame, prev_sample_rate, spk_times, tts_times

    @staticmethod
    def get_kwargs_by_description(
        chunks: Dict[str, Any],
        mode: str,
        gen_variant: str,
        spk_variant: str,
        tts_variant: str,
        generate_name: bool = False,
    ) -> Dict[str, Any] | Tuple[str, Dict[str, Any]]:
        """
        Construct keyword arguments describing a simulated TTS generation setup
        based on chunk timing data and variant specifications.

        :param chunks: Dictionary containing chunk-related arrays:
        :param mode: TTS generation mode, either "parallel" or "sequential".
        :param gen_variant: Generation timing variant:
                            - "full": use real generation times.
                            - "maxed": use constant time equal to the last generation timestamp.
                            - "zero": all generation times set to zero.
        :param spk_variant: Speech playback variant:
                            - "full": use real speech durations.
                            - otherwise: use zeros.
        :param tts_variant: TTS processing variant:
                            - "full": use real TTS times.
                            - otherwise: use zeros.
        :param generate_name: If True, return a (name, kwargs) tuple for logging or identification.
        :returns:
            - If `generate_name=False`: dict containing simulation parameters:
                {
                    "gen_times": np.ndarray,
                    "spk_times": np.ndarray,
                    "tts_times": np.ndarray,
                    "add_tts_in_parrallel": bool,
                }
            - If `generate_name=True`: tuple (setup_name, kwargs_dict)
        """
        gen_times = chunks["gen_times"]
        tts_times = chunks["tts_times"]
        spk_times = chunks["spk_times"]
        name = f"{mode}_gen-{gen_variant}_spk-{spk_variant}_tts-{tts_variant}"

        if gen_variant == "full":
            tmp_gen_times = gen_times
        elif gen_variant == "maxed":
            tmp_gen_times = np.ones_like(gen_times) * gen_times[-1]
        else:
            tmp_gen_times = np.zeros_like(gen_times)
        
        kwargs = {  "gen_times": tmp_gen_times,
                    "spk_times": spk_times if spk_variant == "full" else np.zeros_like(spk_times),
                    "tts_times": tts_times if tts_variant == "full" else np.zeros_like(tts_times),
                    "add_tts_in_parrallel": mode == "parallel",
                }
        if generate_name:
            return name, kwargs
        return kwargs

    def get_chunks_with_tts(self,
        token_times: Sequence[Tuple[str, float, int]],
        k_chunks: int = 5,
        return_audio: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]: 
        """
        Chunk generated tokens (while preserving LaTeX expressions) and generate
        corresponding audio using the TTS pipeline.

        This function:
        1. Splits the token sequence into coherent text chunks.
        2. Converts LaTeX formulas within each chunk to spoken equivalents.
        3. Synthesizes TTS audio for each chunk and records timing information.

        :param token_times: Sequence of (decoded_token, generation_timestamp, step) tuples.
                            Includes the end-of-sequence token.
        :param k_chunks: Target number of tokens per chunk, excluding LaTeX constraints.
        :param return_audio: Whether to return concatenated audio waveform and rate.
        :returns:
            - If `return_audio=False`: dict with per-chunk metadata:
                {
                    "text_chunks": List[str],
                    "chunk_sizes": np.ndarray,
                    "gen_times": np.ndarray,
                    "tts_times": np.ndarray,
                    "spk_times": np.ndarray,
                }
            - If `return_audio=True`: tuple of (chunks_dict, audio_dict), where:
                audio_dict = {
                    "frame": np.ndarray,
                    "frame_rate": int,
                }
        """

        chunked_token_times = self.chunk_tokens_with_latex(token_times[:-1], k=k_chunks)
        texts = [self.convert_markdown_with_latex(el["text"]) for el in chunked_token_times]
        gen_times = [el["times"][-1] for el in chunked_token_times]
        gen_steps = [el["steps"] for el in chunked_token_times]
        chunk_sizes = [len(el["times"]) for el in chunked_token_times]

        total_frame, frame_rate, spk_times, tts_times = self.get_audio_track(texts, k_chunks)

        audio = {
            "frame": total_frame,
            "frame_rate": frame_rate,
            }

        chunks = {
            "text_chunks": texts,
            "chunk_sizes": np.array(chunk_sizes),
            "gen_times": np.array(gen_times), # timestamps
            "gen_steps": gen_steps, # step index
            "tts_times": np.array(tts_times), # timedeltas
            "spk_times": np.array(spk_times), # timedeltas
            }
        
        return (chunks, audio) if return_audio else chunks

    def __call__(self,
        gen_times: Sequence[float],
        spk_times: Sequence[float],
        tts_times: Sequence[float],
        gen_steps: Sequence[Sequence[int]] = None,
        add_tts_in_parrallel: bool = True,
        text_chunks=None,
        chunk_sizes=None,
        return_delays=True,
    ) -> Dict[str, Any]:
        """
        Compute user-perceived delay metrics for a sequence of generated and spoken chunks.

        This function models how generated text chunks are converted into speech,
        accounting for either parallel or sequential TTS generation strategies.

        :param gen_times: List of timestamps when each text chunk finished generation.
        :param spk_times: List of playback durations (seconds) for each chunk.
        :param tts_times: List of TTS generation durations (seconds) for each chunk.
        :param add_tts_in_parrallel: If True, assumes TTS runs in parallel with generation.
                                    If False, models sequential TTS after generation.
        :param text_chunks: Optional list of decoded text chunks (for reference only).
        :param chunk_sizes: Optional list of number of tokens per chunk (for reference only).
        :returns: Dictionary with delay and duration metrics

        :example:
        >>> metrics = evaluator([1.0, 3.0, 6.0], [2.0, 1.5, 3.0], [0.5, 0.8, 1.2])
        >>> metrics["total_delay"]
        0.7
        """
        assert len(gen_times) == len(spk_times), f"{len(gen_times)}, {len(spk_times)}"
        assert len(gen_times) == len(tts_times), f"{len(gen_times)}, {len(tts_times)}"
        if gen_steps is not None:
            assert len(gen_times) == len(gen_steps), f"{len(gen_times)}, {len(gen_steps)}"

        if add_tts_in_parrallel:
            chunk_ready = np.array(gen_times) + np.array(tts_times)
        else:
            chunk_ready = np.array(gen_times) + np.cumsum(tts_times)
        
        delays = np.array(self.compute_delays(chunk_ready, spk_times))
        
        metrics = {
            "delay_to_first": float(delays[0]),
            "total_delay": float(np.sum(delays)),
            "total_delay_mius1": float(np.sum(np.maximum(delays - 1, 0))),
            "duration_no_delay": float(np.sum(spk_times)),
            "duration_with_delay": float(np.sum(spk_times) + float(np.sum(delays))),
        }
        if gen_steps is not None:
            delay_minus10steps = 0
            prev_generated_step = 0
            for chunk in gen_steps:
                for el in chunk:
                    delay_minus10steps += max(el - prev_generated_step - 10, 0)
                    prev_generated_step = el
                    
            metrics.update({
                "steps_to_first": int(gen_steps[0][0]),
                "delay_steps": int(1 + gen_steps[-1][-1] - sum([len(el) for el in gen_steps])),
                "delay_minus10steps": int(delay_minus10steps),
            })
        if return_delays:
            metrics.update({"delays": np.array(delays)})
        return metrics
