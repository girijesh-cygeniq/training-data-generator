from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        """Generate training data from input text."""
        pass

    def _validate_json_response(self, line: str) -> Optional[Dict[str, str]]:
        """Validate and parse JSON response."""
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return None
            if 'instruction' not in data or 'response' not in data:
                return None
            return data
        except json.JSONDecodeError:
            return None