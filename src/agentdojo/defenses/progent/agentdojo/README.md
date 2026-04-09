# AgentDojo

## Installation

```bash
pip install -e . # install agentdojo
cd ..
pip install -e . # install progent
```
Check out more about agentdojo in [README-agentdojo.md](README-agentdojo.md).

## How to Run
```bash
./run.sh
```

## Note
For the injection task 5 in the slack suite, the original implementation directly looks for the attack tool call in the execution trace to determine whether the attack is successful, no matter the result returned by the tool call. With Progent, even if the tool is blocked, it still exists in the trace with a blocking message, and it would be wrongly classified. We manually checked all results for this injection task and corrected the results.
