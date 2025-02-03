# src/llm_providers/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import logging
from src.exceptions import LLMProviderError

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement.
    Each provider (OpenAI, Claude, Ollama) will inherit from this base class.
    """
    
    @abstractmethod
    def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        """Generate training data from input text.
        
        Args:
            text (str): Input text to generate training data from
            
        Returns:
            List[Dict[str, str]]: List of instruction-response pairs
            
        Raises:
            LLMProviderError: If there's an error in generation
        """
        raise NotImplementedError("Each LLM provider must implement generate_training_data method")

    def _validate_json_response(self, line: str) -> Optional[Dict[str, str]]:
        """Validate and parse JSON response.
        
        Args:
            line (str): JSON string to validate
            
        Returns:
            Optional[Dict[str, str]]: Validated JSON object or None if invalid
        """
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return None
            if 'instruction' not in data or 'response' not in data:
                return None
            return data
        except json.JSONDecodeError:
            return None