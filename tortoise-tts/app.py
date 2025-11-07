import os
import torch
import gradio as gr
import torchaudio
import time
from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, load_voices

VOICE_OPTIONS = [
    "angie",
    "deniro",
    "freeman",
    "halle",
    "lj",
    "myself",
    "pat2",
    "snakes",
    "tom",
    "daws",
    "dreams",
    "grace",
    "lescault",
    "weaver",
    "applejack",
    "daniel",
    "emma",
    "geralt",
    "jlaw",
    "mol",
    "pat",
    "rainbow",
    "tim_reynolds",
    "atkins",
    "dortice",
    "empire",
    "kennard",
    "mouse",
    "william",
    "jane_eyre",
    "random",  # special option for random voice
]


def inference(
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

    start_time = time.time()

    # all_parts = []
    for j, text in enumerate(texts):
        for audio_frame in tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset="ultra_fast",
            k=1
        ):
            # print("Time taken: ", time.time() - start_time)
            # all_parts.append(audio_frame)
            yield (24000, audio_frame.cpu().detach().numpy())

    # wav = torch.cat(all_parts, dim=0).unsqueeze(0)
    # print(wav.shape)
    # torchaudio.save("output.wav", wav.cpu(), 24000)
    # yield (None, gr.make_waveform(audio="output.wav",))
def main():
    title = "Tortoise TTS 🐢"
    description = """
    A text-to-speech system which powers lot of organizations in Speech synthesis domain.
    <br/>
    a model with strong multi-voice capabilities, highly realistic prosody and intonation.
    <br/>
    for faster inference, use the 'ultra_fast' preset and duplicate space if you don't want to wait in a queue.
    <br/>
    """
    text = gr.Textbox(
        lines=4,
        label="Text (Provide either text, or upload a newline separated text file below):",
    )
    script = gr.File(label="Upload a text file")

    voice = gr.Dropdown(
        VOICE_OPTIONS, value="jane_eyre", label="Select voice:", type="value"
    )
    voice_b = gr.Dropdown(
        VOICE_OPTIONS,
        value="disabled",
        label="(Optional) Select second voice:",
        type="value",
    )
    split_by_newline = gr.Radio(
        ["Yes", "No"],
        label="Split by newline (If [No], it will automatically try to find relevant splits):",
        type="value",
        value="No",
    )

    output_audio = gr.Audio(label="streaming audio:", streaming=True, autoplay=True)
    # download_audio = gr.Audio(label="dowanload audio:")
    interface = gr.Interface(
        fn=inference,
        inputs=[
            text,
            script,
            voice,
            voice_b,
            split_by_newline,
        ],
        title=title,
        description=description,
        outputs=[output_audio],
    )
    interface.queue().launch()


if __name__ == "__main__":
    tts = TextToSpeech(kv_cache=True, use_deepspeed=True, half=True)

    with open("Tortoise_TTS_Runs_Scripts.log", "a") as f:
        f.write(
            f"\n\n-------------------------Tortoise TTS Scripts Logs, {datetime.now()}-------------------------\n"
        )

    main()