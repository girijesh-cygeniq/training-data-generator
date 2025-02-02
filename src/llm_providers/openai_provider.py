from typing import List, Dict
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseLLMProvider
from src.exceptions import LLMProviderError

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        try:
            response = await openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are an expert in creating training data in JSONL format. Each response must be valid JSON objects with 'instruction' and 'response' fields."
                }, {
                    "role": "user",
                    "content": f"Create training data pairs from this text:\n\n{text}"
                }],
                max_tokens=4000,
                temperature=0.7
            )
            
            jsonl_data = []
            response_text = response.choices[0].message.content.strip()
            
            for line in response_text.split('\n'):
                if data := self._validate_json_response(line.strip()):
                    jsonl_data.append(data)
            
            return jsonl_data
            
        except Exception as e:
            raise LLMProviderError(f"OpenAI API error: {str(e)}")