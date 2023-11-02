import re 
from collections import defaultdict
from nltk import bigrams

def get_word_frequency_map(strings: list, end_token = "</s>") -> defaultdict:
	"""
	takes a list of strings then returns a mapping from tokens (delimited by spaces) to frequencies 
	"""
	word_frequency_map = defaultdict(int)
	for string in strings:
		for word in string.split():
			word_frequency_map[" ".join(list(word)) + " " + end_token] += 1
	return word_frequency_map

def get_bigram_frequency_map(word_frequency_map: defaultdict) -> defaultdict:
	"""
	takes a word frequency map and returns a map of bigram pairs to their frequencies
	"""
	bigrams_frequency_map = defaultdict(int)
	for token_string, frequency in word_frequency_map.items():
		tokens = token_string.split()
		for gram_1, gram_2 in bigrams(tokens):
			bigrams_frequency_map[(gram_1, gram_2)] += frequency
	return bigrams_frequency_map

def merge_into_word_frequency_map(word_frequency_map: defaultdict, bigram_pair: tuple) -> defaultdict:
    """
    takes a word frequency map and a tuple pairing of tokens then replaces all occurance of the pair with a single token

    for example: merge ('t', 'h') into {'t h i s </s>': n} -> {'th i s </s>': n}  
    """
    new_word_frequency_map = defaultdict(int)

    # ('t', 'h') -> 't\ h'
    bigram = re.escape(' '.join(bigram_pair))

    # ensure there is a space delimiter before an after it
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 

    # 't\ h' -> `th`
    replace = ''.join(bigram_pair)
    for token_string in word_frequency_map:
        
        # replace any 't\ h' with 'th' in the token_string
        new_token_string = pattern.sub(replace, token_string) 
        new_word_frequency_map[new_token_string] = word_frequency_map[token_string] 
    return new_word_frequency_map

def byte_pair_encoding(corpus: list, n: int): 
	""" 
	takes a corpus (list of strings) and a integer for the number of merges and returns a word frequence map
	"""
	word_frequency_map = get_word_frequency_map(corpus) 
	for _ in range(n): 
		bigram_frequency_map = get_bigram_frequency_map(word_frequency_map) 
		most_frequent_bigram = max(bigram_frequency_map, key=bigram_frequency_map.get) 
		word_frequency_map = merge_into_word_frequency_map(word_frequency_map, most_frequent_bigram) 
	return word_frequency_map 