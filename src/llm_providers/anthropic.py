from typing import List, Dict
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseLLMProvider
from src.exceptions import LLMProviderError

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        try:
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{
                    "role": "system",
                    "content": "You are an expert in creating training data in JSONL format. Generate instruction-response pairs."
                }, {
                    "role": "user",
                    "content": f"Create training data pairs from this text:\n\n{text}"
                }]
            )
            
            jsonl_data = []
            response_text = response.content[0].text.strip()
            
            for line in response_text.split('\n'):
                if data := self._validate_json_response(line.strip()):
                    jsonl_data.append(data)
            
            return jsonl_data
            
        except Exception as e:
            raise LLMProviderError(f"Anthropic API error: {str(e)}")
