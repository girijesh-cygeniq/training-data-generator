# Training Data Generator

A sophisticated Python application for generating high-quality training data pairs using various Large Language Models (LLMs). This tool helps automate the creation of instruction-response pairs from text or PDF inputs, making it easier to create training datasets for fine-tuning language models.

## Features

- **Multiple Input Formats**
  - Direct text input support
  - PDF file processing with text extraction
  - Bulk processing capabilities

- **LLM Provider Integration**
  - Anthropic's Claude
  - OpenAI's GPT models
  - Local Ollama models
  - Extensible architecture for adding new providers

- **Robust Processing**
  - Automatic retry mechanism for API failures
  - Comprehensive error handling
  - Input validation and sanitization
  - JSON output validation

- **User Interface**
  - Clean and intuitive Gradio web interface
  - Real-time processing feedback
  - Easy provider selection
  - Configurable output format

## Prerequisites

- Python 3.9 or higher
- Valid API keys for Anthropic and OpenAI (if using those providers)
- Ollama installed locally (if using Ollama provider)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/training-data-generator.git
cd training-data-generator
```

2. Create and activate a virtual environment:
```bash
# For Linux/MacOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your API keys:
```env
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Usage

1. Start the application:
```bash
python -m src.main
```

2. Access the web interface:
   - Open your browser and navigate to `http://localhost:7860`
   - The interface will be displayed with three main sections:
     - Text input area
     - PDF file upload
     - LLM provider selection

3. Generate training data:
   - Enter text directly into the text area OR
   - Upload a PDF file
   - Select your preferred LLM provider
   - Click "Submit" to generate training data pairs

4. Output format:
   The generated data will be in JSONL format:
```jsonl
{"instruction": "What is X?", "response": "X is a concept that..."}
{"instruction": "How does Y work?", "response": "Y works by..."}
```

## Project Structure

```
training_data_generator/
├── .env                    # Environment variables
├── requirements.txt        # Project dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── exceptions.py      # Custom exceptions
│   ├── llm_providers/     # LLM provider implementations
│   ├── text_processor.py  # Text processing utilities
│   └── main.py           # Main application entry point
└── tests/                # Test suite
```

## Error Handling

The application handles various error scenarios:
- Invalid API keys
- Network connectivity issues
- PDF processing errors
- Invalid JSON responses
- Rate limiting

Error messages will be displayed in the UI with appropriate context and suggestions for resolution.

## Limitations

- PDF processing may not preserve complex formatting
- API rate limits apply for Claude and OpenAI
- Large PDFs may take longer to process
- Some providers may have token limits

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a Pull Request

## Development

For development and testing:

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Check code style:
```bash
flake8 src/
black src/
```

## Troubleshooting

Common issues and solutions:

1. **API Key Errors**
   - Verify your API keys in the `.env` file
   - Ensure the environment variables are loaded

2. **PDF Processing Issues**
   - Check PDF file permissions
   - Ensure PDF is not encrypted
   - Verify PDF is text-based (not scanned)

3. **Provider Errors**
   - Check network connectivity
   - Verify API rate limits
   - Ensure provider service is available

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Gradio](https://gradio.app/)
- PDF processing using [PyMuPDF](https://pymupdf.readthedocs.io/)
- LLM providers: Anthropic, OpenAI, and Ollama

## Support

For support and questions:
- Open an issue in the GitHub repository
- Contact the maintainers
- Check the documentation

## Security

- API keys are handled securely through environment variables
- Input is sanitized before processing
- Output is validated before returning

## Future Improvements

- Additional LLM provider support
- Batch processing capabilities
- Export options for different formats
- Custom prompt templates
- Advanced PDF processing options
- Performance optimizations
```

