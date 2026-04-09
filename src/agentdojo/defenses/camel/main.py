# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import cyclopts
from agentdojo import attacks, benchmark, logging
from agentdojo.task_suite import get_suite
from openai.types.chat import ChatCompletionReasoningEffort

import camel.custom_yaml  # noqa
from camel.interpreter.interpreter import MetadataEvalMode
from camel.models import make_tools_pipeline


def main(
    model: str,
    use_original: bool = False,
    reasoning_effort: ChatCompletionReasoningEffort = "medium",
    thinking_budget_tokens: int | None = None,
    ad_defense: str | None = None,
    run_attack: bool = False,
    replay_with_policies: bool = False,
    suites: list[str] | None = None,
    eval_mode: MetadataEvalMode = MetadataEvalMode.NORMAL,
    q_llm: str | None = None,
    user_tasks: list[str] | None = None,
):
    """Example usage of the defense.


    Other newer models might work as well.

    Args:
        model: the model to use. it should be {provider}:model_name. For example, "google:gemini-2.5-pro-preview-06-05".
        use_original: whether to use the original model with tool calling API instead of CaMeL
        reasoning_effort: for OpenAI reasoning models. How much the model should reason. Can be "low", "medium", "high".
        thinking_budget_tokens: how many tokens Anthropic reasoning models can use. Note that Anthropic reasoning models are not supported yet.
        ad_defense: whether to use a defense from AgentDojo and which one. It must be used in conjunction with `--use-original`.
            Tested defenses are "tool_filter", "repeat_user_prompt", "spotlight_with_delimiting"
        run_attack: whether to run the attack (it uses AgentDojo's `important_instructions` attack)
        replay_with_policies: replay the run with the given model enforcing security policies. Note that the equivalent run (with same model and attack config)
            should have already been run.
        suites: which suites to run AgentDojo on (can be a list from `["workspace", "banking", "travel", "slack"]`)
        eval_mode: which eval mode to use when propagating dependencies.
        q_llm: what model to use as a quarantined llm. If None, the same as `model` is used.
    """

    attack_name = "important_instructions"

    suites = suites or ["workspace", "banking", "travel", "slack"]
    total_utility_results = []
    total_security_results = []
    logdir = Path("./logs")
    for suite_name in suites:
        tools_pipeline = make_tools_pipeline(
            model,  # type: ignore
            use_original,
            replay_with_policies,
            attack_name,
            reasoning_effort,
            thinking_budget_tokens,
            suite_name,
            ad_defense,
            eval_mode,
            q_llm,  # type: ignore
        )
        suite = get_suite("v1.2", suite_name)
        attack = attacks.load_attack(attack_name, suite, tools_pipeline)
        with logging.OutputLogger(str(logdir)):
            if run_attack:
                results = benchmark.benchmark_suite_with_injections(
                    tools_pipeline,
                    suite,
                    attack,
                    logdir,
                    force_rerun=False,
                    user_tasks=user_tasks,
                )
            else:
                results = benchmark.benchmark_suite_without_injections(
                    tools_pipeline,
                    suite,
                    logdir,
                    force_rerun=False,
                    user_tasks=user_tasks,
                )

        utility_results = results["utility_results"]
        total_utility_results += utility_results.values()
        print(f"{suite_name} - utility: {sum(utility_results.values()) / len(utility_results.values())}")
        if run_attack:
            security_results = results["security_results"]
            total_security_results += security_results.values()
            print(f"{suite_name} - security: {sum(security_results.values()) / len(security_results.values())}")

    print(f"overall - utility: {sum(total_utility_results) / len(total_utility_results)}")
    if run_attack:
        print(f"overall - security: {sum(total_security_results) / len(total_security_results)}")


if __name__ == "__main__":
    cyclopts.run(main)
