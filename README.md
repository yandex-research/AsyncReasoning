# AsyncReasoning


## How to run eval demo:
(The idea is to unify requirements for both hogwild and tts)
1. Update env with newer `./requirements.txt`
2. Install speech-rule-engine (node package) current directory (`.`):
```
# install nodejs >=20.x if not installed already; check with node -v; npm -v
# curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
# sudo apt install -y nodejs

npm init -y
npm install speech-rule-engine
```
3. Run [`./notebooks/demo_simple.ipynb`](./notebooks/demo_simple.ipynb)

4. More detailed notebook for prototyping: [`./notebooks/demo_async_thoughts.ipynb`](./notebooks/demo_async_thoughts.ipynb)

Please make sure that it works for you!

## How to run MATH500 eval

`python -m evals.eval_math-500_split --split-from 0 --split-to 500 --use-slow-kernel --budget 16384 --mode async_reasoning`

## The contents of the `./tortoise` Folder

The code contained in the `./tortoise` directory is **not authored by this project’s maintainers**. It is a replica of the open-source [[Tortoise TTS](https://github.com/neonbjb/tortoise-tts)](https://github.com/neonbjb/tortoise-tts) project by **neonbjb**, which is licensed under the [[Apache License 2.0](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE)](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE). See [(Betker et al, 2023)](https://arxiv.org/abs/2305.07243) for their technical report.

All rights to the original code, documentation, and model files in that directory remain with the original Tortoise TTS authors and contributors. The code is included here only for convenience and compatibility; please refer to the upstream repository for its development history, updates, and license terms.
