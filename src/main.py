# src/main.py

import gradio as gr
import logging
from typing import Optional, Union, Tuple
import json
import tempfile
import os
from datetime import datetime
from src.config import Config
from src.text_processor import TextProcessor
from src.llm_providers import (
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self):
        try:
            config = Config()
            self.providers = {
                "Claude": AnthropicProvider(config.anthropic_api_key),
                "OpenAI": OpenAIProvider(config.openai_api_key),
                "Ollama": OllamaProvider()
            }
            self.text_processor = TextProcessor()
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def _save_jsonl_file(self, data: str) -> str:
        """Save data to a JSONL file and return the file path."""
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(temp_dir, f"training_data_{timestamp}.jsonl")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
                
            return file_path
        except Exception as e:
            logger.error(f"Error saving JSONL file: {str(e)}")
            return None

    def generate_data(
        self,
        text_input: str,
        pdf_file: Optional[Union[bytes, str]] = None,
        provider: str = "Claude"
    ) -> Tuple[str, str]:
        """Generate training data and return both text output and file path."""
        try:
            if not text_input and not pdf_file:
                return "Error: Please provide either text input or upload a PDF file.", None

            # Process input
            processed_text = self.text_processor.process_input(text_input, pdf_file)
            if not processed_text.strip():
                return "Error: No valid text content found.", None

            # Get provider
            provider_instance = self.providers.get(provider)
            if not provider_instance:
                return f"Error: Invalid provider selected: {provider}", None

            # Generate data
            try:
                jsonl_data = provider_instance.generate_training_data(processed_text)
                if not jsonl_data:
                    return "Error: No training data was generated.", None
                    
                # Convert to JSONL format
                jsonl_output = "\n".join(json.dumps(item) for item in jsonl_data)
                
                # Save to file
                file_path = self._save_jsonl_file(jsonl_output)
                if not file_path:
                    return jsonl_output, None
                    
                return jsonl_output, file_path
                
            except Exception as e:
                logger.error(f"Provider error: {str(e)}")
                return f"Error with {provider}: {str(e)}", None
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}", None

def create_ui() -> gr.Interface:
    generator = TrainingDataGenerator()

    def generate_wrapper(*args, **kwargs):
        try:
            output_text, file_path = generator.generate_data(*args, **kwargs)
            if file_path:
                return output_text, file_path
            return output_text, None
        except Exception as e:
            logger.error(f"Error in generate wrapper: {str(e)}")
            return f"Error: {str(e)}", None

    return gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(
                label="Input Text",
                placeholder="Enter text to generate training data...",
                lines=5
            ),
            gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            ),
            gr.Radio(
                choices=["Claude", "OpenAI", "Ollama"],
                label="Select LLM Provider",
                value="Claude"
            )
        ],
        outputs=[
            gr.Textbox(
                label="Generated Training Data (JSONL)",
                lines=10
            ),
            gr.File(
                label="Download Generated Data",
                file_types=[".jsonl"]
            )
        ],
        title="Training Data Generator",
        description="Generate training data pairs from text or PDF input using various LLM providers.",
        examples=[
            ["This is a sample text about AI and machine learning.", None, "OpenAI"]
        ]
    )

if __name__ == "__main__":
    try:
        ui = create_ui()
        ui.launch(share=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise