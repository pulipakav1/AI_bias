import os 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

# API KEYS 
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

#MODELS
CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    
]

OPENAI_MODELS = [
    
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash"
]

PERPLEXITY_MODELS = [
    "sonar",
    "sonar-pro",
]

ALL_MODELS = {
    "claude": CLAUDE_MODELS,
    "openai": OPENAI_MODELS,
    "gemini": GEMINI_MODELS,
    "perplexity": PERPLEXITY_MODELS
}

# Open-Weighted Models
LLAMA31_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct",
]

LLAMA32_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

GEMMA_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

HF_MODELS = {
    "llama31": LLAMA31_MODELS,
    "llama32": LLAMA32_MODELS,
    "gemma": GEMMA_MODELS
}

#REQUIREMENTS
PROMPTS_PER_MODEL = 1000
MAX_TOKENS = 500
OUTPUT_DIR = "results"
PLOTS_DIR = "plots"
TABLES_DIR = "tables"

for dir_name in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(dir_name, exist_ok=True)

#DEMOGRAPHIC AXES 
DEMOGRAPHIC_AXES = [
    "age",
    "disability_status",
    "gender_identity",
    "nationality",
    "physical_appearance",
    "race_ethnicity",
    "race_x_gender",
    "race_x_ses",
    "religion",
    "ses",
    "sexual_orientation"
]