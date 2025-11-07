# AsyncReasoning


## How to run eval demo:
(The idea is to unify requirements for both hogwild and tts)
1. Update env with newer `./requirements.txt`
2. Install speech-rule-engine (node package) current directory (`.`):
```
npm init -y
npm install speech-rule-engine
```
3. Run `demo_with_step_timestamps.ipynb` (same demo with small modifications) that will store some timestamps of token generations
4. Run `./step_eval.ipynb`

Please make sure that it works for you!


# Tortoise-tts readme
---
title: Tortoise Tts
emoji: 🐢
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'ExpressivText-to-Speech '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference