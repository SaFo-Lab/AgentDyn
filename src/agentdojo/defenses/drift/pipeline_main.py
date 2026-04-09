import time
from client import OpenAIModel, OpenRouterModel, GoogleModel

from import_lib import *
from utils import get_args, set_seed, get_logger
from DRIFTLLM import DRIFTLLM
from DRIFTTaskSuite import DRIFTTaskSuite
from DRIFTToolsExecutionLoop import DRIFTToolsExecutionLoop


def main(args, suite_type):
    benchmark_version = args.benchmark_version
    suites = tuple(get_suites(benchmark_version).keys())
    suites = (suite_type,) # banking, slack, travel, workspace

    model_name = args.model

    if args.adaptive_attack:
        output_name = f"{model_name}-adaptive_attack/{suites[0]}"
    else:
        output_name = f"{model_name}/{suites[0]}"

    output_dir = os.path.join("runs", output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Set Attacker
    if args.do_attack:
        attacker = args.attack_type
    else:
        attacker = None

    # if attacker is not None:
    #     save_dir = os.path.join(output_dir, attacker)
    
    # else:
    #     save_dir = os.path.join(output_dir, "user_task")

    save_dir = output_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    logger_path = os.path.join(save_dir, "log.txt")
    logger = get_logger(logger_path)
    logger.info(f"Log File is saved at: {logger_path}")

    logger.info(f"Evaluating Suites: {suites}")

    if model_name.startswith("gpt-"):
        client = OpenAIModel(model=args.model, logger=logger)
        tools_pipeline_name = 'gpt-4o-2024-05-13'
        logger.info(f"Using OpenAI Client: {args.model}")

    elif model_name.startswith("gemini-"):
        client = GoogleModel(model=args.model, logger=logger)
        tools_pipeline_name = args.model
        logger.info(f"Using Google Client: {args.model}")

    else:
        client = OpenRouterModel(model=args.model, logger=logger)
        tools_pipeline_name = args.model
        logger.info(f"Using OpenRouter Client: {args.model}")
        # raise ValueError("Invalid model name.")

    llm = DRIFTLLM(args, client, logger=logger)

    tools_loop = DRIFTToolsExecutionLoop(
        [
            ToolsExecutor(),
            # PromptInjectionDetector(),
            llm,
        ]
    )
    tools_pipeline = AgentPipeline(
        [
            # SystemMessage("You are a helpful agent assistant with superior ."),
            InitQuery(),
            llm,
            tools_loop,
        ]
    )

    for suite_name in suites:
        suite = get_suite(benchmark_version, suite_name)
        task_suite = DRIFTTaskSuite(
            args,
            suite.name,
            suite.environment_type,
            suite.tools,
            suite.data_path,
            suite.benchmark_version,
            parent_instance = suite,
        )

    if args.target_user_tasks is None:
        tasks_to_run = task_suite.user_tasks.values()
        logger.info("Evaluate on all User Tasks.")

    else:
        target_user_task_id = args.target_user_tasks
        tasks_to_run = [task_suite.user_tasks[f"user_task_{task_id}"] for task_id in args.target_user_tasks.split(",")]
        logger.info(f"Evaluate on User Tasks of {target_user_task_id}.")

    utility_result = []
    security_result = []
    tools_pipeline.name = tools_pipeline_name # ['meta-llama/Llama-3-70b-chat-hf', 'gemini-1.5-pro-002', 'claude-3-sonnet-20240229', command-r', 'command-r'-plus, 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', 'mistralai/Mixtral-8x7B-Instruct-v0.1']
    # tools_pipeline.name = "meta-llama/Llama-3-70b-chat-hf"

    resume_utility = 0
    resume_security = 0
    resume_total = 0
    if attacker is not None:
        logger.info(f"Using Attack Method: {attacker}")
        attack = load_attack(attacker, task_suite, tools_pipeline)
        target_injection_tasks = args.target_injection_tasks
        if target_injection_tasks is not None:
            injection_tasks_to_run = {
            injection_task_id: suite.get_injection_task_by_id(injection_task_id)
            for injection_task_id in args.target_injection_tasks.split(",")
            }
            logger.info(f"Injection Tasks of {target_injection_tasks}.")
        else:
            logger.info("Evaluate on all injection tasks.")
            injection_tasks_to_run = task_suite.injection_tasks

        for idx, user_task in enumerate(tasks_to_run):
            user_task_name = user_task.ID
            match = re.fullmatch(r'user_task_(\d+)', user_task_name)
            user_task_idx = int(match.group(1))
            for injec_idx, injection_task_id in enumerate(injection_tasks_to_run):
                match = re.fullmatch(r'injection_task_(\d+)', injection_task_id)
                injection_task_idx = int(match.group(1))
                pre_total_tokens = llm.client.total_tokens

                result_file_path = Path(save_dir) / f"user_task_{user_task_idx}" / attacker / f"injection_task_{injection_task_idx}.json"
                result_file_path.parent.mkdir(parents=True, exist_ok=True)
                if not args.force_rerun and os.path.exists(result_file_path):
                    try:
                        with open(result_file_path, "r", encoding="utf-8") as f:
                            loaded_result = json.load(f)
                            if "utility" in loaded_result and "security" in loaded_result:
                                utility_result.append(loaded_result["utility"])
                                security_result.append(loaded_result["security"])
                                logger.info(f"user_task_{user_task_idx} with injection_task_{injection_task_idx} result already exists; skipping run.\nAttack Success Ratio: {security_result.count(True) + resume_security} / {len(security_result) + resume_total}\nUtility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")
                                continue

                    except Exception as e:
                        logger.info(f"Loading existing result file at {result_file_path} failed: {e}. The task will be re-run.")
                        logger.info(f"Re-runing user_task_{user_task_idx}-injection_task_{injection_task_idx} ...")


                injection_task = suite.get_injection_task_by_id(injection_task_id)
                task_injections = attack.attack(user_task, injection_task)

                start_time = time.time()
                utility, security, messages = task_suite.run_task_with_pipeline(tools_pipeline, user_task, injection_task, task_injections)
                end_time = time.time()
                utility_result.append(utility)
                security_result.append(security)
                with open(result_file_path, "w") as f:
                    json.dump({"suite_name": suite_type, "pipeline_name": f"{args.model}", "user_task_id": f"user_task_{user_task_idx}", "injection_task_id": f"injection_task_{injection_task_idx}", "attack_type": f"{attacker}", "build_constraints": args.build_constraints, "injection_isolation": args.injection_isolation, "dynamic_validation": args.dynamic_validation, "adaptive_attack": args.adaptive_attack, "tool_permission": llm.tool_permissions, "initial_trajectory": llm.initial_function_trajectory, "initial_checklist": llm.initial_node_checklist, "conversations": messages, "benchmark_version": args.benchmark_version, "utility": utility, "security": security, "total_tokens": llm.client.total_tokens - pre_total_tokens, "duration": end_time - start_time}, f, indent=4)

                logger.info(f"user_task_{user_task_idx} with injection_task_{injection_task_idx} Utility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")
                logger.info(f"user_task_{user_task_idx} with injection_task_{injection_task_idx} Attack Success Ratio: {security_result.count(True) + resume_security} / {len(security_result) + resume_total}")

    else:
        logger.info("Evaluating on User Tasks.")
        for idx, user_task in enumerate(tasks_to_run):
            user_task_name = user_task.ID
            match = re.fullmatch(r'user_task_(\d+)', user_task_name)
            user_task_idx = int(match.group(1))
            pre_total_tokens = llm.client.total_tokens
            
            result_file_path = Path(save_dir) / f"user_task_{user_task_idx}" / "none" / f"none.json"
            result_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.force_rerun and os.path.exists(result_file_path):
                try:
                    with open(result_file_path, "r", encoding="utf-8") as f:
                        loaded_result = json.load(f)
                        if "utility" in loaded_result and "security" in loaded_result:
                            utility_result.append(loaded_result["utility"])
                            security_result.append(loaded_result["security"])
                            logger.info(f"user_task_{user_task_idx} result already exists; skipping run.\nAttack Success Ratio: {security_result.count(True) + resume_security} / {len(security_result) + resume_total}\nUtility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")
                            continue

                except Exception as e:
                    logger.info(f"Loading existing result file at {result_file_path} failed: {e}. The task will be re-run.")
                    logger.info(f"Re-runing user_task_{user_task_idx} ...")


            start_time = time.time()
            utility, security, messages = task_suite.run_task_with_pipeline(tools_pipeline, user_task, injection_task=None, injections={})
            end_time = time.time()
            utility_result.append(utility)
            security_result.append(security)
            with open(result_file_path, "w") as f:
                json.dump({"suite_name": suite_type, "pipeline_name": f"{args.model}", "user_task_id": f"user_task_{user_task_idx}", "injection_task_id": None, "attack_type": None, "build_constraints": args.build_constraints, "injection_isolation": args.injection_isolation, "dynamic_validation": args.dynamic_validation, "adaptive_attack": args.adaptive_attack, "tool_permission": llm.tool_permissions, "initial_trajectory": llm.initial_function_trajectory, "initial_checklist": llm.initial_node_checklist, "conversations": messages, "benchmark_version": args.benchmark_version, "utility": utility, "security": security, "total_tokens": llm.client.total_tokens - pre_total_tokens, "duration": end_time - start_time}, f, indent=4)

                logger.info(f"user_task_{user_task_idx} Utility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")

    logger.info(f"Overall Utility Success Ratio: {(utility_result.count(True) + resume_utility) / (len(utility_result) + resume_total)}")
    logger.info(f"Overall Attack Success Ratio: {(security_result.count(True) + resume_security) / (len(security_result) + resume_total)}")
    logger.info(f"{suite_type} Done!")

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    suites = args.suites.split(",")
    for suite_type in suites:
        main(args, suite_type)


    