# Progent: Programmable Privilege Control for LLM Agents
We provide a modular programming framework for expressing fine-grained privilege control policies to secure LLM agents.
Check out our paper [here](https://arxiv.org/abs/2504.11703).

## Installation
```bash
pip install -e .
```
*The documents and demos will be available soon.*

## Experiments in the paper
### Agentdojo
```bash
cd agentdojo
pip install -e . # install agentdojo
cd ..
pip install -e . # install progent
cd agentdojo
./run.sh
```
Check out more in [agentdojo/README.md](agentdojo/README.md)

### ASB
```bash
cd asb
pip install -r requirements.txt # install asb
cd ..
pip install -e . # install progent
cd asb
python scripts/agent_attack.py --cfg_path config/OPI.yml
```
Check out more in [asb/README.md](asb/README.md)

### EhrAgent
```bash
cd ehragent
conda env create -f environment.yml # install ehragent agentpoison
cd ..
pip install -e . # install progent
cd ehragent
./run.sh
```
Check out more in [ehragent/README.md](ehragent/README.md)
