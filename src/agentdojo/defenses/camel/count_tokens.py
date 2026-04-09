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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from camel.count_tokens import count_tokens_for_model

sns.set_style("whitegrid")
sns.set_context("paper")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


DEFENSE_NAMES = {
    "camel": "\\sysname",
    "spotlighting_with_delimiting": "Spotlighting",
    "tool_filter": "Tool Filter",
    "repeat_user_prompt": "Prompt Sandwiching",
}

DEFENSES_TO_PLOT = {"camel"}


def formatter_fn(x_val, tick_pos):
    if int(x_val) == x_val:
        return f"{int(x_val)}x"
    return f"{x_val:.1f}x"


formatter = FuncFormatter(formatter_fn)


def main(model: str, logs_dir: Path = Path("logs"), attack: str | None = None) -> None:
    results_no_defense = count_tokens_for_model(logs_dir, model, attack)
    df_no_defense = pd.DataFrame(results_no_defense)

    results = []

    print("No defense")
    print(
        f"input tokens: mean={np.mean(df_no_defense['input_tokens']):.2f}, median={np.median(df_no_defense['input_tokens'])}, std={np.std(df_no_defense['input_tokens']):.2f}"
    )
    print(
        f"output tokens: mean={np.mean(df_no_defense['output_tokens']):.2f}, median={np.median(df_no_defense['output_tokens'])}, std={np.std(df_no_defense['output_tokens']):.2f}"
    )

    results.append(
        {
            "Defense": "None",
            "Tokens": "Input",
            "Mean": np.mean(df_no_defense["input_tokens"]),
            "Median": np.median(df_no_defense["input_tokens"]),
            "Std": np.std(df_no_defense["input_tokens"]),
        }
    )
    results.append(
        {
            "Defense": "None",
            "Tokens": "Output",
            "Mean": np.mean(df_no_defense["output_tokens"]),
            "Median": np.median(df_no_defense["output_tokens"]),
            "Std": np.std(df_no_defense["output_tokens"]),
        }
    )

    attack_suffix = f"-{attack}" if attack is not None else ""
    filename = f"tokens{attack_suffix}.csv"
    df_no_defense.to_csv(logs_dir / model / filename, index=False)

    increase_results = []

    for defense, defense_name in DEFENSE_NAMES.items():
        defense_name = DEFENSE_NAMES[defense]
        results_defense = count_tokens_for_model(logs_dir, model + f"+{defense}", attack)
        df_defense = pd.DataFrame(results_defense)
        print(f"With {defense_name}")
        print(
            f"input tokens: mean={np.mean(df_defense['input_tokens']):.2f}, median={np.median(df_defense['input_tokens'])}, std={np.std(df_defense['input_tokens']):.2f}"
        )
        print(
            f"output tokens: mean={np.mean(df_defense['output_tokens']):.2f}, median={np.median(df_defense['output_tokens'])}, std={np.std(df_defense['output_tokens']):.2f}"
        )

        results.append(
            {
                "Defense": defense_name,
                "Tokens": "Input",
                "Mean": np.mean(df_defense["input_tokens"]),
                "Median": np.median(df_defense["input_tokens"]),
                "Std": np.std(df_defense["input_tokens"]),
            }
        )
        results.append(
            {
                "Defense": defense_name,
                "Tokens": "Output",
                "Mean": np.mean(df_defense["output_tokens"]),
                "Median": np.median(df_defense["output_tokens"]),
                "Std": np.std(df_defense["output_tokens"]),
            }
        )

        merged_df = pd.merge(
            df_defense,
            df_no_defense,
            on=["suite", "user_task", "injection_task"],
            suffixes=(f"_{defense}", f"_no_{defense}"),
        )

        input_tokens_increase = merged_df[f"input_tokens_{defense}"] / merged_df[f"input_tokens_no_{defense}"]
        output_tokens_increase = merged_df[f"output_tokens_{defense}"] / merged_df[f"output_tokens_no_{defense}"]
        print("Tokens increase (per-task)")
        print(
            f"input tokens increase: mean={np.mean(input_tokens_increase):.2f}, median={np.median(input_tokens_increase):.2f}, std={np.std(input_tokens_increase):.2f}"
        )
        print(
            f"output tokens increase: mean={np.mean(output_tokens_increase):.2f}, median={np.median(output_tokens_increase):.2f}, std={np.std(output_tokens_increase):.2f}"
        )

        increase_results.append(
            {
                "Defense": defense_name,
                "Tokens": "Input",
                "Mean": np.mean(input_tokens_increase),
                "Median": np.median(input_tokens_increase),
                "Std": np.std(input_tokens_increase),
            }
        )

        increase_results.append(
            {
                "Defense": defense_name,
                "Tokens": "Output",
                "Mean": np.mean(output_tokens_increase),
                "Median": np.median(output_tokens_increase),
                "Std": np.std(output_tokens_increase),
            }
        )

        input_tokens_fig = plt.figure(figsize=(4, 3))

        plot_defense_name = defense_name if defense_name != "\\sysname" else "CaMeL"
        ax = sns.histplot(input_tokens_increase, log_scale=True, figure=input_tokens_fig, bins=20)
        xlims = ax.get_xlim()
        ax.set_xlabel(f"tokens with {plot_defense_name} / tokens without {plot_defense_name}\n(input tokens, per task)")
        ax.set_xticks([0.1, 1, 10, 100])
        ax.axvline(float(np.median(input_tokens_increase)), 0.0, 1.0, color="red")
        new_xticks_list = list(ax.get_xticks())
        new_xticks_list.append(float(np.median(input_tokens_increase)))
        ax.set_xticks(sorted(list(set(new_xticks_list))))
        ax.set_xlim(xlims)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_ylabel("")
        if defense in DEFENSES_TO_PLOT:
            input_tokens_fig.savefig(
                f"plots/tokens_increase/input_tokens_increase-{model}-{defense}{attack_suffix}.pdf", bbox_inches="tight"
            )

        output_tokens_fig = plt.figure(figsize=(4, 3))
        ax = sns.histplot(output_tokens_increase, log_scale=True, figure=output_tokens_fig, bins=20)
        ax.set_xlabel(
            f"tokens with {plot_defense_name} / tokens without {plot_defense_name}\n(output tokens, per task)"
        )
        ax.set_xticks([0.1, 1, 10, 100])
        xlims = ax.get_xlim()
        ax.axvline(float(np.median(output_tokens_increase)), 0.0, 1.0, color="red")
        ax.set_ylabel("")
        new_xticks_list = list(ax.get_xticks())
        new_xticks_list.append(float(np.median(input_tokens_increase)))
        ax.set_xticks(sorted(list(set(new_xticks_list))))
        ax.set_xlim(xlims)
        ax.xaxis.set_major_formatter(formatter)
        if defense in DEFENSES_TO_PLOT:
            output_tokens_fig.savefig(
                f"plots/tokens_increase/output_tokens_increase-{model}-{defense}{attack_suffix}.pdf",
                bbox_inches="tight",
            )

        df_defense.to_csv(logs_dir / (model + f"+{defense}") / filename, index=False)

    overall_results_df = pd.DataFrame(results).set_index(["Defense", "Tokens"])
    increase_df = pd.DataFrame(increase_results).set_index(["Defense", "Tokens"])

    overall_results_df.to_latex(
        f"tables/token-usage-{model}{attack_suffix}.tex",
        index=True,
        caption="Token usage by multiple defenses.",
        label=f"tab:token-usage{attack_suffix}",
        column_format="llrrr",
        index_names=True,
        float_format=lambda x: f"${x}$" if isinstance(x, int) else f"${x:.2f}$",
    )

    increase_df.to_latex(
        f"tables/token-increase-{model}{attack_suffix}.tex",
        index=True,
        caption="Token usage increase by multiple defenses.",
        label=f"tab:token-increase{attack_suffix}",
        column_format="llrrr",
        index_names=True,
        float_format=lambda x: f"${x}$" if isinstance(x, int) else f"${x:.2f}$",
    )


if __name__ == "__main__":
    cyclopts.run(main)
