"""
Core functionality for cssallmlib LLM operations
"""

from typing import Optional, Dict, Any
from loguru import logger


class LLMHelper:
    def __init__(self, max_length: int = 2048, template_vars: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM Helper

        Args:
            max_length (int): Maximum allowed length for prompts
            template_vars (Optional[Dict[str, Any]]): Default template variables
        """
        logger.info("Initializing LLM Helper")
        self.max_length = max_length
        self.template_vars = template_vars or {}

    def process_prompt(self, prompt: str, **kwargs) -> str:
        """
        Process and prepare a prompt for LLM consumption

        Args:
            prompt (str): The input prompt to process
            **kwargs: Additional template variables

        Returns:
            str: Processed prompt ready for LLM input

        Raises:
            ValueError: If prompt is empty or exceeds max length
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.debug(f"Processing prompt: {prompt}")
        
        # Apply template variables
        template_vars = {**self.template_vars, **kwargs}
        processed_prompt = prompt.format(**template_vars)

        # Basic sanitization
        processed_prompt = processed_prompt.strip()

        # Check length
        if len(processed_prompt) > self.max_length:
            logger.warning(f"Prompt exceeds maximum length of {self.max_length}")
            processed_prompt = processed_prompt[:self.max_length]

        return processed_prompt

    def set_template_vars(self, vars: Dict[str, Any]) -> None:
        """
        Set or update template variables

        Args:
            vars (Dict[str, Any]): Template variables to set/update
        """
        self.template_vars.update(vars)

    def clear_template_vars(self) -> None:
        """Clear all template variables"""
        self.template_vars.clear()
