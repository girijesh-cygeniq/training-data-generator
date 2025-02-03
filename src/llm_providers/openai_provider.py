# src/llm_providers/openai_provider.py

from typing import List, Dict
import math
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseLLMProvider
from src.exceptions import LLMProviderError

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        """Initialize OpenAI client with API key."""
        self.client = OpenAI(api_key=api_key)
        
    def _calculate_pairs_count(self, text: str) -> int:
        """Calculate minimum number of training pairs based on text length."""
        # Calculate word count (rough approximation)
        word_count = len(text.split())
        
        # Define thresholds
        min_pairs = 5  # Minimum pairs for very short texts
        pairs_per_thousand = 4  # Number of pairs per 1000 words
        
        # Calculate recommended pairs (minimum 5, scales with text length)
        recommended_pairs = max(min_pairs, math.ceil((word_count / 1000) * pairs_per_thousand))
        
        return recommended_pairs
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        """Generate training data using OpenAI's API."""
        try:
            # Calculate minimum pairs based on text length
            min_pairs = self._calculate_pairs_count(text)
            
            prompt = f'''As an AI engineer expert in the cybersecurity domain, carefully read the following text and create training data pairs to fine-tune an LLM.
            The goal is to inject all the knowledge from the given text into the LLM.

            Create instruction-response pairs that cover various aspects of the text, including:
            - Key concepts and definitions
            - Technical details and specifications
            - Use cases and applications
            - Best practices and guidelines
            - Relationships between different components

            Important Guidelines:
            1. Based on the length of the provided text, generate at least {min_pairs} unique instruction-response pairs
            2. Ensure comprehensive coverage of the entire text
            3. Create a mix of:
               - Direct knowledge questions
               - Scenario-based questions
               - Implementation questions
               - Analytical questions
            4. Make responses detailed but concise
            5. Avoid redundant or overlapping questions
            6. Ensure each pair adds unique value

            Text to analyze:
            {text}

            Generate the training pairs in JSONL format, where each line contains a JSON object with "instruction" and "response" fields.
            Remember do not provide any code, provide your response only in {{"instruction":"response"}} JSONL pairs.
            
            You must generate at least {min_pairs} training pairs for this text. If you generate fewer pairs, it will be considered incomplete.'''

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert in creating training data pairs. You must respond only in valid JSONL format with instruction and response fields. Generate at least {min_pairs} unique training pairs. Do not include any explanations or other text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            jsonl_data = []
            response_text = response.choices[0].message.content.strip()
            
            # Debug logging
            print(f"Target pairs: {min_pairs}")
            print("OpenAI raw response:", response_text[:500])
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line:  # Skip empty lines
                    if data := self._validate_json_response(line):
                        jsonl_data.append(data)
            
            # Validate minimum pairs requirement
            if len(jsonl_data) < min_pairs:
                print(f"Warning: Generated only {len(jsonl_data)} pairs, expected {min_pairs}")
                
            if not jsonl_data:
                raise LLMProviderError("No valid JSONL data generated")
            
            return jsonl_data
            
        except Exception as e:
            raise LLMProviderError(f"OpenAI API error: {str(e)}")