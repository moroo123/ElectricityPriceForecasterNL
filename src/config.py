from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration."""
    
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DB_PATH = DATA_DIR / "cache.db"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # External APIs
    ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
    
    # Market Settings
    BIDDING_ZONE = "NL"
    
    # Bidding zones map (for future reference/expansion)
    BIDDING_ZONES = {
        "NL": "Netherlands",
        "BE": "Belgium",
        "DE": "Germany"
    }
    
    # Model Parameters
    DEFAULT_FORECAST_HORIZON = 24
    
    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        if not cls.ENTSOE_API_KEY:
            raise ValueError("ENTSOE_API_KEY environment variable is not set.")
