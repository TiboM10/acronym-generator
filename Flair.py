from flair.embeddings import FlairEmbeddings
language_model = FlairEmbeddings('news-forward').lm
import functools


# higher perplexity = 'worse' grammar
#cache results
@functools.lru_cache(maxsize=1024)
def perplexity(s):
    return language_model.calculate_perplexity(s.lower())
