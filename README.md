# Training Data Generator

A sophisticated Python application for generating high-quality training data pairs using multiple LLM providers (OpenAI, Anthropic Claude, and Ollama). The application processes text or PDF inputs and generates instruction-response pairs in JSONL format.

## Today's Updates (February 3, 2025)

1. Fixed and Optimized LLM Providers:
   - Implemented correct API format for Anthropic Claude
   - Optimized Ollama provider for quality over quantity
   - Standardized OpenAI implementation
   - Added consistent prompt structure across all providers

2. Added Download Functionality:
   - Implemented JSONL file download in UI
   - Added timestamp-based file naming
   - Included proper file handling and cleanup

3. Major Improvements:
   - Fixed response parsing for Claude's TextBlock format
   - Implemented adaptive pair generation based on input length
   - Added robust error handling across providers
   - Enhanced debugging and logging capabilities

## Features

- **Multiple LLM Provider Support:**
  - OpenAI (GPT-3.5/4)
  - Anthropic Claude
  - Ollama (Local LLM)

- **Input Processing:**
  - Direct text input
  - PDF file processing
  - Automatic pair count calculation

- **User Interface:**
  - Clean Gradio web interface
  - Real-time generation
  - File download capability
  - Provider selection

- **Output Formats:**
  - JSONL format output
  - Downloadable files
  - Structured instruction-response pairs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/girijesh-cygeniq/training-data-generator.git
cd training-data-generator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with your API keys:
```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

## Usage

1. Start the application:
```bash
python -m src.main
```

2. Access the web interface:
   - Open your browser to `http://localhost:7860`
   - Enter text or upload a PDF
   - Select your preferred LLM provider
   - Generate training pairs
   - Download the generated data

## Project Structure

```
training_data_generator/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── main.py
│   ├── text_processor.py
│   └── llm_providers/
│       ├── __init__.py
│       ├── base.py
│       ├── anthropic.py
│       ├── openai_provider.py
│       └── ollama_provider.py
├── requirements.txt
└── README.md
```

## LLM Provider Details

### OpenAI Provider
- Uses GPT-3.5/4
- Synchronous implementation
- Adaptive pair generation

### Anthropic Claude Provider
- Uses Claude-3 Sonnet
- Proper TextBlock handling
- Enhanced response parsing

### Ollama Provider
- Local LLM implementation
- Quality-focused generation
- Optimized prompt handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

1. Additional LLM providers
2. More input format support
3. Enhanced data validation
4. Batch processing capabilities
5. Custom prompt templates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Gradio
- Uses OpenAI, Anthropic, and Ollama APIs
- PDF processing using PyMuPDF