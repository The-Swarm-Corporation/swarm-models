"""
Swarms Service Module.

This module provides functionality for interacting with Swarms API.
"""

# Standard library imports
import asyncio
import os
import yaml
from typing import List, Dict

# Third-party imports
from swarms import Agent

# Local imports
from swarm_models.gpt4_vision_api import GPT4VisionAPI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Retrieve the OpenAI API key from environment variables for security.
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the 'OPENAI_API_KEY' environment variable.")


def load_config(file_path: str) -> Dict:
    """Attempt to load a configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Dict: Parsed YAML configuration as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The {file_path} file is missing.") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError("Failed to parse YAML file.") from e


config = load_config("example.yaml")

model = GPT4VisionAPI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o",
    system_prompt=config["system_prompt"],
    logging_enabled=True,
    return_json=False,
)

swarm_reply_agent = Agent(
    llm=model,
    system_prompt=None,
    max_loops=1,
    user_name = "human",
    custom_tools_prompt=None,
)

if __name__ == '__main__':
    """Main function to orchestrate fetching replies for predefined comments."""
    async def main():
        task = "What's this?"
        img_url = "https://pbs.twimg.com/media/GfPLHWNbIAAyk8g?format=jpg&name=medium"
        img_path = "example.png"
        customized_messages = [
            {"role": "system", "content": config["system_prompt"]}
            , {"role": "user", "content": [
                {"type": "text","text": f"{task}"}
                , {"type": "image_url","image_url": {"url": f"{img_url}"}}
                ]
            }
        ]
        try:
            # 1/ call with customized_messages which you compose the details of messages by yourself
            content = swarm_reply_agent.run(messages=customized_messages)
            # 2/ call with task and img_url
            content = swarm_reply_agent.run(task=task, img=img_url)
            # 3/ call with task and image_path
            content = swarm_reply_agent.run(task=task, img=img_path)
            # 4/ call with task and multi_imgs, multi_imgs is a list of image_path or image_url
            content = swarm_reply_agent.run(task=task, multi_imgs=[img_url, img_path])
        except Exception as e:
            print(f"An error occurred: {e}")

    asyncio.run(main())