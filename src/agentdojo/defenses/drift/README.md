# DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents

[Hao Li](https://leolee99.github.io/), [Xiaogeng Liu](https://sheltonliu-n.github.io/), [Hung-Chun Chiu](https://qhjchc.notion.site/), [Dianqi Li](https://scholar.google.com/citations?user=K40nbiQAAAAJ&hl=en), [Ning Zhang](https://cybersecurity.seas.wustl.edu/index.html), [Chaowei Xiao](https://xiaocw11.github.io/).

<p align="center" width="80%">
<a target="_blank"><img src="assets/framework.png" alt="framework" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

The official implementation of NeurIPS 2025 paper "[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](https://www.arxiv.org/pdf/2506.12104)".

## Update
- [2026.1.30] 🛠️ Support the evaluation on more agents.
- [2026.1.30] 🛠️ Update the evaluation code on ASB.


## How to Start
We provide the evaluation of DRIFT, you can reproduce the results following:

### Construct Your Environment

```bash
conda create -n drift python=3.11
source activate drift
pip install "agentdojo==0.1.35"
pip install -r requirements.txt
```

### Set Your API KEY
We provide three API providers, including OpenAI, Google, and OpenRouter. Please set up the API Key as you need.

```bash
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
```

### run task with no attack
```bash
python pipeline_main.py \
--model gpt-4o-mini-2024-07-18 \
--build_constraints --injection_isolation --dynamic_validation
```

### run task under attack
```bash
python pipeline_main.py \
--model gpt-4o-mini-2024-07-18 --do_attack \
--attack_type important_instructions \
--build_constraints --injection_isolation --dynamic_validation
```

You can evaluate any model from the supported providers by passing its model identifier (eg., gemini-2.5-pro) to the `--model` flag. To evaluate under an adaptive attack, include the `--adaptive_attack` configuration.


### Evaluating on ASB
Please refer to `ASB_DRIFT/README.md`.


### Inspect Results
You can find the cached results in `runs/`.


## References

If you find this work useful in your research or applications, we appreciate that if you can kindly cite:

```
@articles{DRIFT,
  title={DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents},
  author={Hao Li and Xiaogeng Liu and Hung-Chun Chiu and Dianqi Li and Ning Zhang and Chaowei Xiao},
  journal = {NeurIPS},
  year={2025}
}
```
