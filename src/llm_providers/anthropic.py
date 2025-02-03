# src/llm_providers/anthropic_provider.py

from typing import List, Dict
import math
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseLLMProvider
from src.exceptions import LLMProviderError

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        """Initialize Anthropic client with API key."""
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def _calculate_pairs_count(self, text: str) -> int:
        """Calculate minimum number of training pairs based on text length."""
        word_count = len(text.split())
        min_pairs = 5
        pairs_per_thousand = 4
        return max(min_pairs, math.ceil((word_count / 1000) * pairs_per_thousand))

    def _clean_response_text(self, response_text: str) -> str:
        """Clean the response text from any prefixes or formatting."""
        # Remove common prefixes that Claude might add
        prefixes_to_remove = [
            "Here are the training pairs in JSONL format:",
            "Here are the training pairs:",
            "Here are",
            "Training pairs:",
        ]
        
        clean_text = response_text
        for prefix in prefixes_to_remove:
            if clean_text.startswith(prefix):
                clean_text = clean_text[len(prefix):].strip()
        
        # Remove any leading/trailing whitespace and newlines
        return clean_text.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        """Generate training data using Claude."""
        try:
            min_pairs = self._calculate_pairs_count(text)
            
            system_prompt = (
                f"You are an expert in creating training data pairs. "
                f"Generate at least {min_pairs} unique training pairs in JSONL format. "
                f"Respond only with valid JSON objects having 'instruction' and 'response' fields. "
                f"Do not include any prefixes, suffixes, or explanations."
            )

            prompt = (
                f"As an AI engineer expert in the cybersecurity domain, carefully read the following text "
                f"and create training data pairs to fine-tune an LLM. "
                f"The goal is to inject all the knowledge from the given text into the LLM.\n\n"
                f"Create instruction-response pairs that cover various aspects of the text, including:\n"
                f"- Key concepts and definitions\n"
                f"- Technical details and specifications\n"
                f"- Use cases and applications\n"
                f"- Best practices and guidelines\n"
                f"- Relationships between different components\n\n"
                f"Important Guidelines:\n"
                f"1. Based on the length of the provided text, generate at least {min_pairs} unique instruction-response pairs\n"
                f"2. Ensure comprehensive coverage of the entire text\n"
                f"3. Create a mix of:\n"
                f"   - Direct knowledge questions\n"
                f"   - Scenario-based questions\n"
                f"   - Implementation questions\n"
                f"   - Analytical questions\n"
                f"4. Make responses detailed but concise\n"
                f"5. Avoid redundant or overlapping questions\n"
                f"6. Ensure each pair adds unique value\n\n"
                f"Text to analyze:\n{text}\n\n"
                f'Generate ONLY valid JSON objects, one per line. Each line must be in the format: {{"instruction":"question","response":"answer"}}\n'
                f"Do not include any explanations or additional text."
            )

            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Get the actual response text
            response_text = message.content[0].text
            print(f"Initial response: {response_text[:200]}")
            
            # Clean the response text
            clean_response = self._clean_response_text(response_text)
            print(f"Cleaned response: {clean_response[:200]}")
            
            jsonl_data = []
            for line in clean_response.split('\n'):
                line = line.strip()
                if line:  # Skip empty lines
                    if data := self._validate_json_response(line):
                        jsonl_data.append(data)
                        print(f"Valid pair found: {line[:100]}")
            
            if not jsonl_data:
                print(f"Failed to parse any valid JSON from response. Full response:\n{response_text}")
                raise LLMProviderError("No valid JSONL data in response")

            print(f"Successfully generated {len(jsonl_data)} pairs")
            return jsonl_data
            
        except Exception as e:
            print(f"Error in generate_training_data: {str(e)}")
            raise LLMProviderError(f"Claude API error: {str(e)}")