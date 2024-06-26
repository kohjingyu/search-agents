"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import collections
import copy
import glob
import heapq
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import openai
import requests
import torch
from PIL import Image

from agent import (
    PromptAgent,
    construct_agent,
    value_function
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent, create_goto_url_action
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt", choices=["prompt", "search"])
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # search config
    parser.add_argument("--max_depth", type=int, default=4, help="Max depth for search agents.")
    parser.add_argument("--branching_factor", type=int, default=5, help="Branching factor at each step for the search agent.")
    parser.add_argument("--search_algo", type=str, default="vf", help="Search algorithm to use", choices=["vf", "bfs", "dfs"])
    parser.add_argument("--vf_budget", type=int, default=20, help="Budget for the number of value function evaluations.")
    parser.add_argument("--value_function", type=str, default="gpt4o", help="What value function to use.", choices=["gpt4o"])

    # example config
    parser.add_argument("--test_idx", type=str, default=None, help="Idx to test")
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    config_file_list: list[str]
) -> None:
    scores = []
    max_steps = args.max_steps
    branching_factor = args.branching_factor
    assert args.vf_budget is not None, "Value function budget should be specified."

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == 'visualwebarena':
        if (
            caption_image_fn
            and args.eval_captioning_model == args.captioning_model
        ):
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    for config_file in config_file_list:
        render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )
        try:
            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            input_image = Image.open(requests.get(image_path, stream=True).raw)
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            action_history = []  # Save the action history for the agent so that we can backtrack.
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info, "url": env.page.url}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            step_idx = 0
            while True:
                step_idx += 1
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                            branching_factor=branching_factor
                        )
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")
                
                # BEGIN SEARCH
                if args.agent_type == "search" and type(action) == list:
                    evaluator = evaluator_router(
                        config_file, captioning_fn=eval_caption_image_fn
                    )
                    best_score, best_actions = None, []
                    all_candidates = []
                    all_inputs = []  # For input to the value function computation.

                    def take_action_and_score(a, curr_trajectory, curr_action_history,
                                              curr_obs_metadata, last_screenshots: list[Image.Image], 
                                              generate_next_actions: bool = True, depth: int = 0,
                                              branching_factor: int = 5, a_idx: int = -1, curr_a_idx: int = -1):
                        """Take the given actions and score the resulting trajectory."""
                        temp_trajectory = copy.deepcopy(curr_trajectory)
                        temp_action_history = copy.deepcopy(curr_action_history)
                        should_generate_next_actions = generate_next_actions
                        img_after_path = f"{task_id}_step{step_idx}_action{a_idx}_depth{depth}_curra{curr_a_idx}_after.png"
                        obs, _, terminated, _, info = env.step(a)
                        # Save the image after the action.
                        obs_img = Image.fromarray(obs["image"])
                        all_inputs.append({
                            "text": obs["text"],
                            "image": obs["image"],
                            "image_after_path": img_after_path,
                            "action": a["raw_prediction"],
                        })
                        temp_trajectory.append(a)

                        # Get natural language description of current action
                        curr_action_str = get_action_description(
                            a, curr_obs_metadata, action_set_tag=args.action_set_tag,
                            prompt_constructor=agent.prompt_constructor if isinstance(agent, PromptAgent) else None,
                        )
                        temp_action_history.append(curr_action_str)

                        if a["action_type"] == ActionTypes.STOP:
                            should_generate_next_actions = False
                        elif a["action_type"] != ActionTypes.STOP:
                            temp_trajectory.append({"observation": obs, "info": info, "url": env.page.url})
                        elif terminated:
                            should_generate_next_actions = False
                            temp_trajectory.append(create_stop_action(""))
                        
                        start_time = time.time()
                        # Only evaluate terminating trajectories
                        try:
                            if args.value_function in ["gpt4o"]:
                                score = value_function.evaluate_success(
                                    screenshots=last_screenshots[-(args.max_depth+1):] + [obs_img], actions=temp_action_history,
                                    current_url=env.page.url, last_reasoning=a["raw_prediction"],
                                    intent=intent, models=["gpt-4o-2024-05-13"],
                                    intent_images=images if len(images) > 0 else None)
                            else:
                                raise NotImplementedError(f"Value function {args.value_function} not implemented")
                        except Exception as e:
                            print(f"Error in evaluator: {e}")
                            score = 0

                        next_actions = []
                        if score < 1 and should_generate_next_actions:
                            start_time = time.time()
                            temp_early_stop_flag, _ = early_stop(
                                temp_trajectory, max_steps, early_stop_thresholds
                            )
                            if not temp_early_stop_flag:
                                try:
                                    # Generate possible action candidates for next step.
                                    next_actions = agent.next_action(
                                        temp_trajectory,
                                        intent,
                                        images=images,
                                        meta_data=meta_data,
                                        branching_factor=branching_factor
                                    )
                                except ValueError as e:
                                    # get the error message
                                    print('Failed to generate next actions:', e)

                        return score, temp_trajectory, temp_action_history, next_actions
                    
                    def maybe_update_best_action(score, actions):
                        nonlocal best_score, best_actions
                        # The second if statement checks that for scores < 1 that are tied, we should take the one that doesn't terminate.
                        if (best_score is None) or (score > best_score):
                            best_score, best_actions = score, actions
                    
                    max_depth = args.max_depth  # Lookahead of trajectories of length (max_depth + 1)
                    action_queue = []  # Store tuple of (score, a_idx, action, trajectory, depth, ...)
                    actions_at_depth = collections.defaultdict(int)
                    search_counter = 0
                    for a_idx, a in enumerate(action):
                        # Default the score of the initial actions to 0.5.
                        item = (-0.5, a_idx, -1, search_counter, [a], copy.deepcopy(trajectory), copy.deepcopy(meta_data["action_history"]), 0)
                        if args.search_algo == "bfs":
                            action_queue.append(item)
                        elif args.search_algo == "dfs":
                            action_queue.append(item)
                        elif args.search_algo == "vf":
                            heapq.heappush(action_queue, item)
                    
                    while action_queue and search_counter < args.vf_budget:
                        if args.search_algo == "bfs":
                            item = action_queue.pop(0)
                        elif args.search_algo == "dfs":
                            item = action_queue.pop(-1)
                        elif args.search_algo == "vf":
                            item = heapq.heappop(action_queue)
                        curr_score, a_idx, curr_a_idx, _, curr_actions, curr_trajectory, curr_action_history, curr_depth = item

                        search_counter += 1
                        next_action = curr_actions[-1]
                        actions_at_depth[curr_depth] += 1
                        assert len(curr_actions) == curr_depth + 1, f"(depth+1) should be equal to the number of actions taken, but got {len(curr_actions)} and {curr_depth+1}"

                        if next_action["action_type"] == ActionTypes.NONE:
                            maybe_update_best_action(0, curr_actions)
                        else:
                            last_screenshots = []
                            # Reset environment to prepare for next action.
                            _ = env.reset(options={"config_file": config_file})
                            # Take all the previous actions to get back to the current state.
                            start_time = time.time()
                            for a_hist in action_history:
                                obs, _, _, _, info = env.step(a_hist)
                            last_screenshots.append(Image.fromarray(obs["image"]))
                            # Take all previous actions in the current trajectory.
                            start_time = time.time()
                            for a in curr_actions[:-1]:
                                obs, _, _, _, info = env.step(a)
                                last_screenshots.append(Image.fromarray(obs["image"]))  # 1e-5s
                            
                            # Take the next action to evaluate.
                            start_time = time.time()
                            score, new_trajectory, new_action_history, next_actions = take_action_and_score(
                                next_action, curr_trajectory, curr_action_history,
                                copy.deepcopy(info["observation_metadata"]), last_screenshots=last_screenshots[-4:],
                                generate_next_actions=(curr_depth < max_depth),
                                depth=curr_depth, branching_factor=branching_factor, a_idx=a_idx, curr_a_idx=curr_a_idx)
                            raw_pred = next_action['raw_prediction'].split('\n')[-1]
                            all_candidates.append(f'a_idx={a_idx},curr_a_idx={curr_a_idx},depth={curr_depth}: {next_action["raw_prediction"]} (score: {score}, time: {time.time() - start_time})')
                            maybe_update_best_action(score, curr_actions)
                            # Try for next action (if allowed)
                            if score == 1:
                                break
                            else:
                                # Add next actions to the queue.
                                for na_idx, na in enumerate(next_actions):
                                    item = (-score, a_idx, na_idx, search_counter, curr_actions + [na], copy.deepcopy(new_trajectory),copy.deepcopy(new_action_history), curr_depth + 1)
                                    if args.search_algo == "vf":
                                        heapq.heappush(action_queue, item)
                                    else:
                                        action_queue.append(item)
                else:
                    all_candidates = []
                    best_actions = [action]
                    best_score = None

                stop_trajectory = False
                if args.agent_type == "search":
                    # Reset environment to the actual current state to prepare for taking the best action.
                    _ = env.reset(options={"config_file": config_file})
                    prev_url = env.page.url
                    truncated_action_history = []
                    for a_hist in action_history:
                        _ = env.step(a_hist)
                        curr_url = env.page.url
                        # Optimization to simplify the action history, since we will commit the best action.
                        truncated_action_history.append(a_hist)
                        if curr_url != prev_url:
                            # URL has changed, update the truncated_action_history
                            truncated_action_history = [create_goto_url_action(curr_url)]
                            prev_url = curr_url
                    action_history = truncated_action_history

                prev_url = env.page.url
                # Now we can actually execute the best action.
                for best_idx, action in enumerate(best_actions):
                    all_candidates.append(f"Selected action {best_idx}: {action['raw_prediction']}")
                    trajectory.append(action)

                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                    )
                    render_helper.render(
                        action, state_info, meta_data, args.render_screenshot, all_candidates if args.agent_type == "search" else None
                    )

                    meta_data["action_history"].append(action_str)

                    if action["action_type"] == ActionTypes.STOP:
                        stop_trajectory = True
                        break

                    obs, _, terminated, _, info = env.step(action)
                    # Save the committed action to the action history.
                    action_history.append(action)
                    curr_url = env.page.url
                    if curr_url != prev_url:
                        # URL has changed, simplify the action_history so that we resume from this checkpoint
                        action_history = [create_goto_url_action(curr_url)]
                        prev_url = curr_url
                    state_info = {"observation": obs, "info": info, "url": env.page.url}
                    trajectory.append(state_info)

                    if terminated:
                        # add a action place holder
                        trajectory.append(create_stop_action(""))
                        stop_trajectory = True
                        break

                # We solved the task and can quit.
                if stop_trajectory or (best_score is not None and best_score == 1.0):
                    # Save obs
                    break
            # END SEARCH

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )
        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    if args.test_idx is not None:
        print(f"Testing on {args.test_idx}")
        for x in args.test_idx.split(","):
            test_file_list.append(os.path.join(test_config_base_dir, f"{x}.json"))
    else:
        print(f"Testing on {args.test_start_idx} to {args.test_end_idx}")
        st_idx = args.test_start_idx
        ed_idx = args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, test_file_list)
