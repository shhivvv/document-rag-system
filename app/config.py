import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # API Configuration
    API_PORT: int = Field(default=8000, validation_alias="API_PORT")
    API_HOST: str = Field(default="0.0.0.0", validation_alias="API_HOST")

    # Groq Configuration
    GROQ_API_KEY: str = Field(default="", validation_alias="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")

    # Embeddings Configuration
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_NAME")

    # Storage Directories
    CHROMA_PERSIST_DIR: str = Field(default="./data/vector_db", validation_alias="CHROMA_PERSIST_DIR")
    UPLOAD_DIR: str = Field(default="./data/documents", validation_alias="UPLOAD_DIR")

    # Settings config to read from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        if not path.is_absolute():
            path = self.BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_chroma_path(self) -> Path:
        path = Path(self.CHROMA_PERSIST_DIR)
        if not path.is_absolute():
            path = self.BASE_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

# Initialize settings
settings = Settings()
