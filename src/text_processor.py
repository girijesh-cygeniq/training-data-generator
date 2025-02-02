from typing import Union, Optional
from io import BytesIO
import logging
import pymupdf
from .exceptions import TextProcessingError

logger = logging.getLogger(__name__)

class TextProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            stream = BytesIO(pdf_bytes)
            doc = pymupdf.open(stream=stream, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise TextProcessingError(f"Error extracting text from PDF: {str(e)}")

    @staticmethod
    def process_input(text_input: str, pdf_file: Optional[Union[bytes, str]] = None) -> str:
        """Process input text or PDF file."""
        if pdf_file:
            if isinstance(pdf_file, str):
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
            else:
                pdf_bytes = pdf_file
            return TextProcessor.extract_text_from_pdf(pdf_bytes)
        return text_input