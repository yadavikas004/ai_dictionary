from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os

load_dotenv()  # Load environment variables from .env file

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    SECRET_KEY: str = os.getenv("SECRET_KEY")

    class Config:
        env_file = ".env"  # This line is optional if you load it manually

settings = Settings()
