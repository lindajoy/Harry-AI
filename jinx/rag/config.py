import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")

CHROMA_PATH = "chroma"
DATA_PATH = "data/book_1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL = "models/embedding-001"

# Set env at module load
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
