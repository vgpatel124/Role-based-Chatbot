import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    GROQ_API_KEY: str = Field(default="")
    JWT_SECRET_KEY: str = Field(default="finsolve-secret-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=30)
    CHROMA_DB_PATH: str = Field(default="./chroma_db")
    RESOURCES_PATH: str = Field(default="./resources")


settings = Settings(
    GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
    JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY", "finsolve-secret-key-change-in-production"),
    JWT_ALGORITHM=os.getenv("JWT_ALGORITHM", "HS256"),
    JWT_EXPIRE_MINUTES=int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
    CHROMA_DB_PATH=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
    RESOURCES_PATH=os.getenv("RESOURCES_PATH", "./resources"),
)
