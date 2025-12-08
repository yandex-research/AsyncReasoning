# AsyncReasoning


## How to run eval demo:
(The idea is to unify requirements for both hogwild and tts)
1. Update env with newer `./requirements.txt`
2. Install speech-rule-engine (nodejs package, requires node >=14.6) in current directory (`.`):
```
npm init -y
npm install speech-rule-engine
```
3. Run [`./notebooks/demo_simple.ipynb`](./notebooks/demo_simple.ipynb)

4. More detailed notebook for prototyping: [`./notebooks/demo_async_thoughts.ipynb`](./notebooks/demo_async_thoughts.ipynb)

Please make sure that it works for you!

## How to run MATH500 eval

**Single GPU**
```bash
python -m evals.math500 --start 0 --end 500 --mode async_reasoning --path-to-results ./eval_results/math500 --budget 16384 --use-slow-kernel
```

**GPU-parallel**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 utils/gpu_parallel.py --start 0 --end 500 --use_queue --script evals/math500.py \
  --extra_args "--mode async_reasoning --budget 16384 --path-to-results ./eval_results/math500"
```

- `--use-slow-kernel` uses PyTorch code instead of CUDA kernels, it can affect real-time delays.
- `CUDA_VISIBLE_DEVICES` must be specified for `gpu_parallel.py`

## The contents of the `./tortoise` Folder

The code contained in the `./tortoise` directory is **not authored by this project’s maintainers**. It is a replica of the open-source [[Tortoise TTS](https://github.com/neonbjb/tortoise-tts)](https://github.com/neonbjb/tortoise-tts) project by **neonbjb**, which is licensed under the [[Apache License 2.0](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE)](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE). See [(Betker et al, 2023)](https://arxiv.org/abs/2305.07243) for their technical report.

All rights to the original code, documentation, and model files in that directory remain with the original Tortoise TTS authors and contributors. The code is included here only for convenience and compatibility; please refer to the upstream repository for its development history, updates, and license terms.
