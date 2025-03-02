"""
Core functionality for cssallmlib LLM operations
"""

from loguru import logger


class LLMHelper:
    def __init__(self):
        logger.info("Initializing LLM Helper")

    def process_prompt(self, prompt: str) -> str:
        """
        Process and prepare a prompt for LLM consumption

        Args:
            prompt (str): The input prompt to process

        Returns:
            str: Processed prompt ready for LLM input
        """
        logger.debug(f"Processing prompt: {prompt}")
