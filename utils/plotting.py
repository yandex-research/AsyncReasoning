import numpy as np
import matplotlib.pyplot as plt
from itertools import product

possible_colors = [
    'green', 'blue', 'red', 'orange',
    'purple', 'brown', 'pink', 'cyan',
    'magenta', 'olive', 'teal', 'gold',
    'gray', 'navy', 'lime', 'coral',
    'indigo', 'maroon', 'turquoise', 'darkgreen']

def plot_one_timeline(spk_times, delay, color="red"):
    x = np.arange(len(spk_times))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})

    speech_with_delay = np.cumsum(spk_times + delay)
    ax1.plot(x, speech_with_delay, linestyle="-", color=color, label=f'Speach with delays')
    ax1.plot(x, speech_with_delay - delay, linestyle="--", color=color)
    ax1.fill_between(
        x, speech_with_delay - delay, speech_with_delay,
        color=color,
        alpha=0.2,
        label="Delay gap"
    )
    ax2.hist(delay[delay > 0], bins=20, color=color, alpha=0.4, edgecolor="black")


    # Ideal scenario when llm, tts finish immidietly, but human hears with certain speed
    speech_no_delay = np.cumsum(spk_times)
    ax1.plot(x, speech_no_delay, linestyle="--", color="black",label='Speech without delays')

    ax1.set_xlabel('Chunk index')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Generation and Speech Timeline')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    ax2.set_xlabel("Delay duration (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Delay Duration Distribution")


    plt.tight_layout()
    plt.show()


def plot_timelines(evaluator, chunks, 
    mode_variants = ["parallel", "sequential"],
    gen_variants = ["maxed", "full", "zeroed"],
    spk_variants = ["full", "zeroed"],
    tts_variants = ["full", "zeroed"],
):
    setups = {}

    for mode, gen_variant, spk_variant, tts_variant in product(
            mode_variants, gen_variants, spk_variants, tts_variants):

        name, kwargs = evaluator.get_kwargs_by_description(
            chunks, mode, gen_variant, spk_variant, tts_variant, generate_name=True)
        setups[name] = kwargs

    spk_times = chunks["spk_times"]
    x = np.arange(len(spk_times))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})


    for (setup_name, kwargs), color in zip(setups.items(), possible_colors[:len(setups)]):
        delay = evaluator(**kwargs)["delays"]

        setup_name = setup_name.replace("_", " ")
        speech_with_delay = np.cumsum(spk_times + delay)
        ax1.plot(x, speech_with_delay, linestyle="-", color=color, label=f'{setup_name}')
        ax1.plot(x, speech_with_delay - delay, linestyle="--", color=color)
        ax1.fill_between(
            x, speech_with_delay - delay, speech_with_delay,
            color=color,
            alpha=0.2,
        )
        ax2.hist(delay[delay > 0], bins=20, color=color, alpha=0.4) # , edgecolor="black"

    ax1.set_xlabel('Chunk index')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Generation and Speech Timeline')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    ax2.set_xlabel("Delay duration (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Delay Duration Distribution")


    plt.tight_layout()
    plt.show()