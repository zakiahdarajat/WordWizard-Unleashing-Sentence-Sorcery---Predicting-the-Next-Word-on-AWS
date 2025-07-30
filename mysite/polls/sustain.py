from django.core.cache import cache
from django.conf import settings
import pickle
import os
import logging

logger = logging.getLogger(__name__)

model_cache_key = 'model_cache'
# this key is used to `set` and `get` your trained model from the cache

tokenizer = cache.get(model_cache_key) # get tokenizer from cache

if tokenizer is None:
    # your model isn't in the cache
    # so `set` it
    # load the tokenizer with absolute path
    try:
        tokenizer_path = os.path.join(settings.BASE_DIR, 'polls', 'tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        cache.set(model_cache_key, tokenizer, None)
        logger.info("Tokenizer loaded successfully from file")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        tokenizer = None
