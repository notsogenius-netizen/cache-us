"""
Configuration module for loading environment variables.
"""
import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    cerebras_api_key: str
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    deepgram_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: Optional[str] = None
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: Optional[str] = None
    twilio_webhook_base_url: str

    # Database
    db_url: str

    # Vector Database (FAISS)
    vector_db_choice: str = "faiss"
    faiss_index_dir: str = "./data/faiss_indices"
    embedding_dimension: int = 384

    # Models
    llm_model: str = "gpt-oss-120b"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"

    # WebSocket Configuration
    ws_max_connections: int = 100
    session_timeout: int = 300
    ws_heartbeat_interval: int = 30

    # Audio Streaming Configuration
    audio_sample_rate: int = 8000
    audio_channels: int = 1
    audio_encoding: str = "pcm_mulaw"

    # Agent Configuration
    default_agent_namespace_prefix: str = "agent_"
    max_tools_per_agent: int = 20
    max_knowledge_base_size_mb: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

