import os
import json
import logging
from io import BytesIO
from typing import List, Dict, Union

import pymupdf  # use the modern PyMuPDF import
from anthropic import Anthropic
import openai
import ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API clients
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
# Ollama is initialized per request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self):
        self.prompt_template = (
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
        )

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF file (provided as bytes) using PyMuPDF."""
        try:
            stream = BytesIO(pdf_bytes)
            # Open the PDF document using pymupdf (modern API)
            doc = pymupdf.open(stream=stream, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    async def generate_with_claude(self, text: str) -> List[Dict[str, str]]:
        """Generate training data using Anthropic's Claude."""
        try:
            response = await anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": self.prompt_template.format(text=text)
                }]
            )
            jsonl_data = []
            for line in response.content[0].text.strip().split('\n'):
                if line.strip():
                    jsonl_data.append(json.loads(line))
            return jsonl_data
        except Exception as e:
            logger.error(f"Error generating with Claude: {str(e)}")
            raise Exception(f"Error generating with Claude: {str(e)}")

    async def generate_with_openai(self, text: str) -> List[Dict[str, str]]:
        """Generate training data using OpenAI."""
        try:
            enhanced_prompt = (
                self.prompt_template +
                "\n\nIMPORTANT: Ensure each line is a valid JSON object containing 'instruction' and 'response' fields. "
                "Example format:\n{'instruction': 'What is X?', 'response': 'X is Y.'}"
            )
            response = await openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in creating high-quality training data in JSONL format. Each response should be properly formatted JSON objects."},
                    {"role": "user", "content": enhanced_prompt.format(text=text)}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            logger.info("This is the response text: ", response_text )
            jsonl_data = []
            
            # First, try parsing line by line assuming proper JSONL output.
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # Only append if both expected keys exist
                    if all(key in obj for key in ['instruction', 'response']):
                        jsonl_data.append(obj)
                    else:
                        logger.warning(f"Skipping JSON object without expected keys: {obj}")
                except json.JSONDecodeError:
                    continue
            
            # If nothing valid was found, use a fallback: split by double newlines and try colon splitting.
            if not jsonl_data:
                logger.info("Fallback parsing: attempting to split response by double newlines.")
                pairs = response_text.split('\n\n')
                for pair in pairs:
                    if ':' in pair:
                        parts = pair.split(':', 1)
                        jsonl_data.append({
                            'instruction': parts[0].strip(),
                            'response': parts[1].strip()
                        })
            
            if not jsonl_data:
                # Log the raw response for debugging purposes.
                logger.error(f"Raw OpenAI response did not contain expected keys:\n{response_text}")
                raise Exception("Could not parse valid training data from OpenAI response.")
            
            return jsonl_data
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise Exception(f"Error generating with OpenAI: {str(e)}")


    def generate_with_ollama(self, text: str) -> List[Dict[str, str]]:
        """Generate training data using Ollama."""
        try:
            response = ollama.generate(
                model='llama2',
                prompt=self.prompt_template.format(text=text),
                stream=False,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 4000,
                }
            )
            response_text = response['response']
            jsonl_data = []
            current_json = ""
            in_json = False
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('{'):
                    in_json = True
                    current_json = line
                elif in_json and line.endswith('}'):
                    current_json += line
                    try:
                        obj = json.loads(current_json)
                        if 'instruction' in obj and 'response' in obj:
                            jsonl_data.append(obj)
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    current_json = ""
                elif in_json:
                    current_json += line
            if not jsonl_data:
                pairs = response_text.split('\n\n')
                for pair in pairs:
                    if ':' in pair:
                        parts = pair.split(':', 1)
                        jsonl_data.append({
                            'instruction': parts[0].strip(),
                            'response': parts[1].strip()
                        })
            return jsonl_data
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            raise Exception(f"Error generating with Ollama: {str(e)}")

    def _get_input_text(self, text_input: str, pdf_file: Union[None, bytes, str]) -> str:
        """
        Retrieve input text either directly or from an uploaded PDF file.
        If pdf_file is provided and is not raw bytes, it is assumed to be a file path and will be read in binary mode.
        """
        if pdf_file:
            # If pdf_file is already bytes, use it directly.
            if isinstance(pdf_file, bytes):
                pdf_bytes = pdf_file
            # If pdf_file is a string (or NamedString), assume it's a file path.
            elif isinstance(pdf_file, str):
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
            # If pdf_file has a read() attribute (like a file object), use that.
            elif hasattr(pdf_file, "read"):
                pdf_bytes = pdf_file.read()
            else:
                raise Exception("Invalid type for pdf_file input.")
            return self.extract_text_from_pdf(pdf_bytes)
        return text_input

    async def _generate_training_data(self, text: str, llm_choice: str) -> List[Dict[str, str]]:
        """Generate training data using the selected LLM provider."""
        if llm_choice == "Claude":
            return await self.generate_with_claude(text)
        elif llm_choice == "OpenAI":
            return await self.generate_with_openai(text)
        else:  # Ollama
            return self.generate_with_ollama(text)

    async def process_input(
        self,
        text_input: str,
        pdf_file: Union[None, bytes, str],
        llm_choice: str
    ) -> str:
        """Process input and generate training data."""
        try:
            input_text = self._get_input_text(text_input, pdf_file)
            if not input_text.strip():
                return "Please provide either text input or upload a PDF file."
            jsonl_data = await self._generate_training_data(input_text, llm_choice)
            jsonl_output = "\n".join(json.dumps(item) for item in jsonl_data)
            return jsonl_output
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return f"Error processing input: {str(e)}"
