from __future__ import annotations
from collections import Counter
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def _split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:

    """
    Split the text into chunks that are separated by special tokens.
    The special tokens are included in the output list.
    """
    if not special_tokens:
        return [text]
    
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    return re.split(pattern, text)

def _pretokenize(text: str, 
                 special_tokens: list[str],
                 ) -> Counter[tuple[bytes, ...]]:
    """
    Pre-tokenize the text into subword units.
    The text is first split into chunks that are separated by special tokens.
    Each chunk is then tokenized into subword units using the regex pattern defined in PAT.
    """
    
    counts: Counter[tuple[bytes, ...]] = Counter()
    
    segments = _split_on_special_tokens(text, special_tokens)
    
    for segment in segments:
        for match in re.finditer(PAT, segment):
            pretoken = match.group(0)
            encoded = pretoken.encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in encoded)
            
            counts[token_tuple] += 1
            
    return counts

def _count_pairs(pretoken_counts: dict[tuple[bytes, ...], int]) -> Counter[tuple[bytes, bytes]]:
    
    """
    Count the frequency of adjacent byte pairs in the pre-tokenized text.
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for tokens, freq in pretoken_counts.items():
        for i in range(len(tokens) - 1):
            pair =  (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq
            
    return pair_counts

def _merge_pair_in_tokens(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Merge the specified pair of bytes in the given token sequence.
    """
    left, right = pair
    merged_token = left + right
    
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right):
            new_tokens.append(merged_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
        
    return tuple(new_tokens)

def _apply_merge(pretoken_counts: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes],) -> dict[tuple[bytes, ...], int]:
   """
   Apply one merge operation to all pre-tokens.
   """
   
   new_counts: Counter[tuple[bytes, ...]]  = Counter()
   
   for tokens, freq in  pretoken_counts.items():
       new_tokens = _merge_pair_in_tokens(tokens, pair)
       new_counts[new_tokens] += freq
    
       
   return dict(new_counts)

def _build_pair_to_words(words: list[tuple[bytes, ...]]) -> dict[tuple[bytes, bytes], set[int]]:
    """
    Build a mapping from byte pairs to the indices of words that contain them.
    """
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}
    
    for word_id, tokens in enumerate(words):
        local_pairs = set(zip(tokens[:-1], tokens[1:]))
        
        for pair in local_pairs:
            
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
                
            pair_to_words[pair].add(word_id)
        
    return pair_to_words

def _apply_merge_incremental(best_pair: tuple[bytes, bytes], words: list[tuple[bytes, ...]], frequencies: list[int], pair_counts: Counter[tuple[bytes, bytes]], pair_to_words: dict[tuple[bytes, bytes], set[int]]) -> None:
    """
    Apply one BPE merge only to words that contain best_pair.
    """
    
    affected_word_ids = list(pair_to_words.get(best_pair, []))
    for word_id in affected_word_ids:
        old_tokens = words[word_id]
        freq = frequencies[word_id]
        
        old_local_pairs = Counter(zip(old_tokens[:-1], old_tokens[1:]))
        new_tokens = _merge_pair_in_tokens(old_tokens, best_pair)
        words[word_id] = new_tokens
        
        new_local_pairs = Counter(zip(new_tokens[:-1], new_tokens[1:]))
        changed_pairs = (set(old_local_pairs) | set(new_local_pairs))
        
        for pair in changed_pairs:
            old_count = old_local_pairs.get(pair, 0)
            new_count = new_local_pairs.get(pair, 0)
            delta = (new_count - old_count) * freq
            
            pair_counts[pair] += delta
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
                
        old_pair_set = set(old_local_pairs)
        new_pair_set = set(new_local_pairs)

        for pair in old_pair_set - new_pair_set:

            pair_to_words[pair].discard(word_id)

            if not pair_to_words[pair]:
                del pair_to_words[pair]

        for pair in new_pair_set - old_pair_set:

            if pair not in pair_to_words:
                pair_to_words[pair] = set()

            pair_to_words[pair].add(word_id)
                
            

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE model on the input text file.
    Returns the vocabulary and the list of merges.
    """
    
    # initialize the 256 single-byte tokens
    vocab: dict[int, bytes] = {
        i: bytes([i])
        for i in range(256)
    }
    
    # add the special tokens to the vocabulary
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
            
    if vocab_size < len(vocab):
        raise ValueError(f"Vocabulary size {vocab_size} is smaller than the number of special tokens {len(vocab)}")
    
    # read the input file and pre-tokenize it
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pretoken_counts = _pretokenize(text, special_tokens)
    words = list(pretoken_counts.keys())
    frequencies = [pretoken_counts[word] for word in words]
    
    pair_counts = _count_pairs(pretoken_counts)
    pair_to_words = _build_pair_to_words(words)
    
    merges: list[tuple[bytes, bytes]] = []
    
    
    # bpe merge loop
    while len(vocab) < vocab_size:
        # pair_counts = _count_pairs(pretoken_counts)
        if not pair_counts:
            break
        
        max_freq = max(pair_counts.values())
        best_pair = max(pair for pair, freq in pair_counts.items() if freq == max_freq)
        
        merges.append(best_pair)
        
        merged_token = (best_pair[0] + best_pair[1])
        vocab[len(vocab)] = merged_token
        
        # pair_counts = _count_pairs(pretoken_counts)
        _apply_merge_incremental(best_pair, words, frequencies, pair_counts, pair_to_words)
        
        
    return vocab, merges
        
        
    
   
    

