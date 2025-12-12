# AsyncReasoning

This repository contains supplementary code for the paper **Asynchronous Reasoning: Training-Free Interactive Thinking LLMs**

More detailed instructions will be added shortly. However, you should be able to run the method itself with the current version.

## How to run eval demo (Qwen3-32B requires a 80GB GPU, can be adapted to 48GB):

1. Create a new python environment (conda / venv) - we tested python 3.14.
2. Install dependencies: `pip install -r ./requirements.txt`.
  - Note that `torch` and `deepspeed` may require custom installation, depending on your hardware.
4. [Optional, but recommended] compile fast GPU kernels:
  - Edit `./inference_lib/pyproject.toml` --- find the following 4 variables: `CUDA_TOOLKIT_ROOT_DIR, CUDA_INCLUDE_DIRS, CUDA_HOME, CMAKE_CUDA_COMPILER`
  - If you are using a custom cuda installation, uncomment `CUDA_TOOLKIT_ROOT_DIR` and point it to the base path of your CUDA installation (e.g. `/usr/local/cuda-12.8`)
  - If you are using cuda toolkit installed with anaconda, un-comment all four variables and set them to paths based on your conda installation.
  - Compile: ```cd inference_lib && pip install -e . && cd ..```
4. Install `speech-rule-engine` (uses nodejs, requires node>=14.6 installed):
  - ```npm init -y```   - **important:** this command should run in the repository root, not in `/`.
  - ```npm install speech-rule-engine```
5. Run the demo: [`./demo_simple.ipynb`](./demo_simple.ipynb) - this is the minimal notebook that illustrates the method interactively.
  - If you skipped compiling the kernels on step 3, set `use_fast_kernel=False` in the demo!
  - If TTSEvaluator fails to compile, see tortoise TTS installation guide: https://github.com/neonbjb/tortoise-tts

If you have questions or encounter difficulties with our code specifically, please open an issue.

## Running MATH500 and MMLU-Pro evals

**Single GPU MATH-500**
```bash
python -m evals.math500 --start 0 --end 500 --mode async_reasoning --path-to-results ./eval_results/math500 --budget 16384 # if did not compile: add --use-slow-kernel
```

**GPU-parallel MATH-500**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 utils/gpu_parallel.py --start 0 --end 500 --use_queue --script evals/math500.py \
  --extra_args "--mode async_reasoning --budget 16384 --path-to-results ./eval_results/"
```

**GPU-parallel MMLU-Pro**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 utils/gpu_parallel.py --start 0 --end 500 --use_queue --script evals/mmlu-pro.py \
  --extra_args "--mode async_reasoning --budget 16384 --num-samples 500 --path-to-results ./eval_results/"
```
Notes:
- `--use-slow-kernel` uses PyTorch code instead of CUDA kernels if you skipped it during installation. Slow kernels will affect real-time delays!
- `CUDA_VISIBLE_DEVICES` must be specified for `gpu_parallel.py` to work.

The results can be aggregated from the `json` files under `./eval_results` folder. For instance:
```bash
cd ./eval_results/math500/async_reasoning
python -c "import os, sys, json; fs=[fn for fn in os.listdir('.') if fn.endswith('.json')]; total_delay = sum(json.load(open(fn))['metrics']['total_delay'] for fn in fs) / len(fs); print(f'{total_delay=:.5f}', file=sys.stderr)"
```

## Safety Experiments

TBU

## The contents of the `./tortoise` Folder

The code contained in the `./tortoise` directory is **not authored by this project’s maintainers**. It is a replica of the open-source [[Tortoise TTS](https://github.com/neonbjb/tortoise-tts)](https://github.com/neonbjb/tortoise-tts) project by **neonbjb**, which is licensed under the [[Apache License 2.0](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE)](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE). See [(Betker et al, 2023)](https://arxiv.org/abs/2305.07243) for their technical report.

All rights to the original code, documentation, and model files in that directory remain with the original Tortoise TTS authors and contributors. The code is included here only for convenience and compatibility; please refer to the upstream repository for its development history, updates, and license terms.
