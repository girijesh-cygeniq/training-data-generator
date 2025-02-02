import gradio as gr
from training_data_generator import TrainingDataGenerator

def create_ui() -> gr.Interface:
    generator = TrainingDataGenerator()
    return gr.Interface(
        fn=generator.process_input,
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
        description="Generate training data pairs from text or PDF input using various LLM providers.",
        examples=[
            ["This is a sample text about cybersecurity...", None, "Claude"],
            ["", "sample.pdf", "OpenAI"]
        ]
    )

if __name__ == "__main__":
    # To create a public link, set share=True below.
    ui = create_ui()
    ui.launch(share=True)
