import os
from dotenv import load_dotenv
from typing import Dict, Any

class Config:
    def __init__(self):
        load_dotenv()
        self.validate_env()
        
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    def validate_env(self) -> None:
        """Validate required environment variables."""
        required_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")