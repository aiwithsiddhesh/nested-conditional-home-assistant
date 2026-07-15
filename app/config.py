from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_model: str

    kitchen_manual_path: Path = Path("data/manuals/kitchen.pdf")
    laundry_manual_path: Path = Path("data/manuals/laundry.pdf")
    hvac_manual_path: Path = Path("data/manuals/hvac.pdf")
    policy_doc_path: Path = Path("data/manuals/policy.pdf")

    chunk_size: int = 800
    chunk_overlap: int = 100
    retriever_k: int = 4


settings = Settings()
