# AsyncReasoning


## How to run eval demo:
(The idea is to unify requirements for both hogwild and tts)
1. Update env with newer `./requirements.txt`
2. Install speech-rule-engine (node package) from `./tortoise-tts` dir:
```
npm init -y
npm install speech-rule-engine
```
3. Run `demo_with_step_timestamps.ipynb` (same demo with small modifications) that will store some timestamps of token generations
4. Run `./tortois-tts/eval.ipynb`

Please make sure that it works for you!