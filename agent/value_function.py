from typing import Tuple, Optional
import base64
from io import BytesIO
import os
import re
from openai import OpenAI
from browser_env.utils import pil_to_b64, pil_to_vertex
import numpy as np
from PIL import Image
import requests
import re

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def evaluate_success(screenshots: list[Image.Image], actions: list[str], current_url: str, last_reasoning: str,
                     intent: str, models: list[str], intent_images: Optional[Image.Image] = None,
                     n: int = 20, top_p: float = 1.0, should_log: bool = False) -> float:
    """Compute the value of a state using the value function.

    Args:
        state (str): The state to compute the value of.
        action (list[str]): The action to take in the state.
        intent (str): The intent to compute the value of.
        intent_images (list[Image.Image], optional): The images corresponding to the intent. Defaults to None.
        file_prefix (str, optional): The prefix to use for the file name. Defaults to ''.
    Returns:
        float: The value of the state.
    """
    last_actions_str = '\n'.join(actions[:-1])
    last_response = actions[-1]
    if intent_images is None:
        content = []
        for screenshot in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "detail": "high"
                },
            })

        content.append({
            "type": "text",
            "text": f"""User Intent: {intent}
Action History: {last_actions_str}
Bot response to the user: {last_response}
Current URL: {current_url}
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
"""
        })

    else:
        content = []

        for img in intent_images:
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(img)
                    },
                }
            ])
        content.append({
            "type": "text",
            "text": f"\nUser Intent: {intent}\n"
        })

        for screenshot in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "detail": "high"
                },
            })

        content.append({
            "type": "text",
            "text": f"""
Action History: {last_actions_str}
Bot response to the user: {last_response}
Current URL: {current_url}
The images corresponding to the user intent are shown in the FIRST {len(intent_images)} images (before the User Intent).
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the LAST {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
"""
        })

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]

    all_responses = []
    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=256,
            top_p=top_p,
            n=n // len(models)
        )
        all_responses.extend(response.choices)

    if should_log:
        print('=' * 30)
        print("Value function input:", content[-1])
    all_scores = []
    for r_idx, r in enumerate(all_responses):
        if should_log:
            print(f"Output {r_idx}: {r.message.content}")
        try:
            pred = re.search(r'Status: "?(.+)"?', r.message.content).group(1)
            if 'success' in pred.lower():
                score = 1.0
            else:
                # Check if it's on the path to success
                on_path = re.search(r'On the right track to success: "?(.+)"?', r.message.content).group(1)
                if 'yes' in on_path.lower():
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0
        
        all_scores.append(score)
    
    score = np.mean(all_scores)
    if should_log:
        print(f"Final score: {score}")
        print('=' * 30)
    return score

