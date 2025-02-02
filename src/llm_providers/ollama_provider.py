from ollama import chat 
from typing import List, Dict
from .base import BaseLLMProvider
from src.exceptions import LLMProviderError

class OllamaProvider(BaseLLMProvider):

     
    def generate_training_data(self, text: str) -> List[Dict[str, str]]:
        prompt = '''
            "As an AI engineer expert in the cybersecurity domain, carefully read the following text and create training data pairs to fine-tune an LLM. "
            "The goal is to inject all the knowledge from the given text into the LLM.\n\n"
            "Create instruction-response pairs that cover various aspects of the text, including:\n"
            "- Key concepts and definitions\n"
            "- Technical details and specifications\n"
            "- Use cases and applications\n"
            "- Best practices and guidelines\n"
            "- Relationships between different components\n\n"
            "Text to analyze:\n{text}\n\n"
            "Generate the training pairs in JSONL format, where each line contains a JSON object with \"instruction\" and \"response\" fields. Make the pairs diverse and comprehensive."
            "Remember donot provide any code, provide your response only in {'instruction':'response', 'instruction':'response', 'instruction':'response'} JSONLpairs.
        '''
        print("This is the prompt given to ollama: ", prompt[:100])
        try:
            response = chat(
                model='llama3.2:3b',
                messages=[
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
            print("Here is the resonse text we got : ", response_text)
            for line in response_text.split('\n'):
                if data := self._validate_json_response(line.strip()):
                    jsonl_data.append(data)
            
            return jsonl_data
            
        except Exception as e:
            raise LLMProviderError(f"Ollama error: {str(e)}")