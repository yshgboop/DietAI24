import os

API_TYPE = "openai"
MODELS = {
    'llm': 'gpt-5-mini',
    'llm_vision': 'gpt-5-mini',  
    'embedding': 'text-embedding-3-large'
}

API_KEYS = {
    'openai': os.environ.get('OPENAI_API_KEY', '')  # Read from environment for security
}