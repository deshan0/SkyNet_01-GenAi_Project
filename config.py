import os

# API Keys - Replace with your actual keys
OPENAI_API_KEY = "your-openai-key-here"  # Replace with actual key
GEMINI_API_KEY = "your-gemini-key-here"  # Replace with actual key

# Data collection settings
SAMPLES_PER_MODEL = 20
DATA_DIR = "data"

# Model settings
LOCAL_MODEL_PATH = "models/rpg_model_final"
CONFIDENCE_THRESHOLD = 0.7

# Generation settings
DEFAULT_THEMES = ["dungeon", "forest", "castle", "cave"]
DEFAULT_DIFFICULTIES = ["easy", "medium", "hard"]
DEFAULT_SIZES = [(15, 10), (20, 15), (25, 20)]