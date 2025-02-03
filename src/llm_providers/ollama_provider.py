# src/llm_providers/ollama_provider.py

from typing import List, Dict
from ollama import chat
from .base import BaseLLMProvider
from src.exceptions import LLMProviderError

class OllamaProvider(BaseLLMProvider):
    def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        """Generate training data using Ollama with focus on quality over quantity."""
        try:
            prompt = '''As an AI engineer expert in the cybersecurity domain, carefully read the following text and create high-quality training data pairs to fine-tune an LLM.
            The goal is to inject the most important knowledge from the given text into the LLM.

            Create focused instruction-response pairs that cover key aspects of the text, including:
            - Key concepts and definitions
            - Technical details and specifications
            - Use cases and applications
            - Best practices and guidelines
            - Relationships between different components

            Important Guidelines:
            1. Focus on quality over quantity
            2. Ensure each pair is highly relevant and accurate
            3. Make responses detailed and precise
            4. Avoid redundant or superficial questions
            5. Cover the most important concepts thoroughly

            Text to analyze:
            {text}

            Generate the training pairs in JSONL format, where each line contains a JSON object with "instruction" and "response" fields.
            Remember do not provide any code, provide your response only in {{"instruction":"response"}} JSONL pairs.'''

            print("Sending request to Ollama...")
            
            response = chat(
                model='llama3.2:3b',
                messages=[
                    {
                        'role': 'system',
                        'content': "You are an expert in creating high-quality training data pairs. Focus on generating precise and accurate JSONL format pairs."
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 4000,
                }
            )
            
            jsonl_data = []
            response_text = response['message']['content']
            print("Response received from Ollama")

            for line in response_text.split('\n'):
                line = line.strip()
                if line:  # Skip empty lines
                    if data := self._validate_json_response(line):
                        jsonl_data.append(data)
            
            if not jsonl_data:
                raise LLMProviderError("No valid JSONL data generated")
            
            print(f"Successfully generated {len(jsonl_data)} high-quality training pairs")
            return jsonl_data
            
        except Exception as e:
            raise LLMProviderError(f"Ollama error: {str(e)}")