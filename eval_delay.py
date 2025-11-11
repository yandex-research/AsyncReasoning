import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import transformers

from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, load_voices

import re
import latex2mathml.converter
import subprocess

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.DEBUG)


class TTSEvaluator:
    def __init__(self):
        self.tts = TextToSpeech(kv_cache=True, use_deepspeed=True, half=True)

    def inference(
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
                k=1
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

        for token_id, tok, t in token_times:
            stripped = tok.lstrip()  # remove leading spaces for detection

            if not inside_math:
                if stripped.startswith('$$'):
                    inside_math = True
                    delimiter = '$$'
                elif stripped.startswith('$'):
                    inside_math = True
                    delimiter = '$'
            else:
                if delimiter == '$$' and '$$' in stripped:
                    inside_math = False
                    delimiter = None
                elif delimiter == '$' and ('$' in stripped and not stripped.startswith('$$')):
                    inside_math = False
                    delimiter = None

            current_tokens.append(tok)
            current_times.append(t)

            if len(current_tokens) >= k and not inside_math:
                flush()

        flush()
        return chunks

    @staticmethod
    def analyze_speech_timing(gen_times, spk_times, show=True, measure_in="chunks", chunk_sizes=None):
        assert len(gen_times) == len(spk_times), f"{len(gen_times)}, {len(spk_times)}"
        assert not (measure_in == "tokens" and chunk_sizes is None), "Provide chunk sizes to plot tokens"

        delays = []
        ideal_starts = []
        actual_starts = []
        shift = 0.0
        total_delay = 0.0
        total_gen_time = gen_times[-1]

        for i, (t_gen, t_speak) in enumerate(zip(gen_times, spk_times)):
            ideal_start = sum(spk_times[:i]) + shift
            ideal_starts.append(ideal_start)
            actual_starts.append(t_gen)
            if t_gen > ideal_start:
                delay = t_gen - ideal_start
                delays.append(delay)
                shift += delay
                total_delay += delay
            else:
                delays.append(0.0)

        delays = np.array(delays)
        speech_no_delay = np.sum(spk_times)
        speech_with_delay = speech_no_delay + total_delay

        metrics = {
            "total_delay": float(total_delay),
            "speech_no_delay": float(speech_no_delay),
            "speech_with_delay": float(speech_with_delay),
            "avg_delay": float(np.mean(delays)),
            "max_delay": float(np.max(delays)),
            "num_delayed_chunks": int(np.sum(delays > 0)),
            "total_num_chunks": int(len(delays)),
        }

        if show:
            if measure_in == "chunks":
                x = np.arange(len(gen_times))

            elif measure_in == "tokens":
                token_positions = np.cumsum([0] + chunk_sizes[:-1])  # start index of each chunk
                total_tokens = sum(chunk_sizes)
                x = token_positions
            else:
                raise ValueError(f"measure_in should be in ['chunks', 'tokens']")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

            # Left: timeline comparison
            ax1.plot(x, ideal_starts, label="ideal start", color="green")
            ax1.plot(x, actual_starts, label="actual start", color="blue")
            ax1.fill_between(
                x, ideal_starts, actual_starts,
                where=(np.array(actual_starts) > np.array(ideal_starts)),
                color="red", alpha=0.3, label="delay region"
            )
            sc = ax1.scatter(
                x, np.array(actual_starts),
                c=delays, cmap="Reds", s=30, label="delay magnitude"
            )

            # Plot speech progression (no delay vs with delay)
            cumulative_no_delay = np.cumsum(spk_times)
            cumulative_with_delay = cumulative_no_delay + np.cumsum(delays)

            ax1.plot(x, cumulative_no_delay, linestyle="--", color="orange", label="speech (no delay)")
            ax1.plot(x, total_gen_time + cumulative_no_delay, linestyle="--", color="orange")
            ax1.plot(x, cumulative_with_delay, linestyle=":", color="black", label="speech (with delay)")

            fig.colorbar(sc, ax=ax1, label="Delay (s)")
            ax1.set_xlabel(f"{measure_in} index")
            ax1.set_ylabel("Time (s)")
            ax1.set_title("Speech Generation Timing Analysis")
            ax1.legend()

            # Right: histogram
            nonzero_delays = delays[delays > 0]
            ax2.hist(nonzero_delays, bins=20, color="red", alpha=0.6, edgecolor="black")
            ax2.set_xlabel("Delay duration (s)")
            ax2.set_ylabel("Count")
            ax2.set_title("Delay Duration Distribution")

            plt.tight_layout()
            plt.show()

        return metrics
    
    def get_audio_track(self, text):
        # !!! Sometimes this code fails due to matrix dim mismatch (that is something wrong with tortois-tts). Just rerun cell.
        # Here is temporary solution
        flag = True
        while flag:
            try:
                frames_srate = []
                spk_times = []
                for sample_rate, frame in self.inference(
                    text=text,
                    script=None,
                    voice="freeman",
                    voice_b="disabled",
                    seed=42,
                    split_by_newline="Yes",
                ):
                    spk_times.append(len(frame) / sample_rate)
                    frames_srate.append((frame, sample_rate))
                flag = False
            except RuntimeError as RE:
                logger.debug(f"Caught RuntimeError: {RuntimeError}")
                print(f"Caught RuntimeError: {RuntimeError}")

        total_frame = np.concatenate([el[0] for el in frames_srate], axis=0)
        return total_frame, frames_srate[0][1], spk_times

    def __call__(self, token_times, k_cunks=5, show=False, measure_in="tokens"):
        """
        Here will be better doc string
        token_times: List[(token_id, decoded_str, generated_timestamp)]
                        ^-- eos included
        """
        chunked_token_times = self.chunk_tokens_with_latex(token_times[:-1], k=k_cunks)

        texts = [el["text"] for el in chunked_token_times]
        gen_times = [el["times"][-1] for el in chunked_token_times]

        text = self.convert_markdown_with_latex("\n".join(texts))
        total_frame, frame_rate, spk_times = self.get_audio_track(text)

        chunk_sizes = [len(el["times"]) for el in chunked_token_times]
        metrics = self.analyze_speech_timing(gen_times, spk_times, show=show, measure_in=measure_in, chunk_sizes=chunk_sizes)
        return metrics, (gen_times, spk_times, chunk_sizes), (total_frame, frame_rate)
