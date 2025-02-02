


class LLMProviderError(Exception):
    """Base exception for LLM provider errors.
    
    This exception is raised when there are issues with LLM providers like OpenAI, 
    Claude, or Ollama. Examples include:
    - API authentication failures
    - Rate limit exceeded
    - Network connectivity issues
    - Invalid responses from the provider
    
    Attributes:
        message (str): The error message
        provider (str, optional): The name of the LLM provider
        status_code (int, optional): HTTP status code if applicable
    """
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(self.message)


class TextProcessingError(Exception):
    """Exception for text processing errors.
    
    This exception is raised when there are issues processing input text or PDF files.
    Examples include:
    - PDF parsing failures
    - Invalid file formats
    - Text encoding issues
    - Memory issues with large files
    
    Attributes:
        message (str): The error message
        file_name (str, optional): Name of the file being processed
        file_size (int, optional): Size of the file in bytes
    """
    def __init__(self, message: str, file_name: str = None, file_size: int = None):
        self.message = message
        self.file_name = file_name
        self.file_size = file_size
        super().__init__(self.message)


class ValidationError(Exception):
    """Exception for validation errors.
    
    This exception is raised when there are issues with input or output validation.
    Examples include:
    - Invalid JSON formatting
    - Missing required fields
    - Data type mismatches
    - Value range violations
    
    Attributes:
        message (str): The error message
        field (str, optional): The field that failed validation
        value (any, optional): The invalid value
    """
    def __init__(self, message: str, field: str = None, value: any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


