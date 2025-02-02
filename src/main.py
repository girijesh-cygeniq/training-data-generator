import gradio as gr
import asyncio
import logging
import json 
from typing import Optional, Union
# Local imports
from src.config import Config
from src.text_processor import TextProcessor
from src.llm_providers import (
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider
)
from src.exceptions import (
    LLMProviderError,
    TextProcessingError,
    ValidationError
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self):
        config = Config()
        self.providers = {
            "Claude": AnthropicProvider(config.anthropic_api_key),
            "OpenAI": OpenAIProvider(config.openai_api_key),
            "Ollama": OllamaProvider()
        }
        self.text_processor = TextProcessor()

    async def generate_data(
        self,
        text_input: str,
        pdf_file: Optional[Union[bytes, str]] = None,
        provider: str = "Claude"
    ) -> str:
        try:
            # Process input
            processed_text = self.text_processor.process_input(text_input, pdf_file)
            if not processed_text.strip():
                return "Please provide either text input or upload a PDF file."

            # Generate training data
            provider_instance = self.providers.get(provider)
            if not provider_instance:
                return f"Invalid provider selected: {provider}"

            jsonl_data = provider_instance.generate_training_data(processed_text)
            return "\n".join(json.dumps(item) for item in jsonl_data)

        except Exception as e:
            logger.error(f"Error generating training data: {str(e)}")
            return f"Error: {str(e)}"

def create_ui() -> gr.Interface:
    generator = TrainingDataGenerator()

    # Wrapper for async function
    def generate_wrapper(*args, **kwargs):
        return asyncio.run(generator.generate_data(*args, **kwargs))

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
        outputs=gr.Textbox(
            label="Generated Training Data (JSONL)",
            lines=10
        ),
        title="Training Data Generator",
        description="Generate training data pairs from text or PDF input using various LLM providers."
    )

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=True)