from collections import Counter

# Computer Term Frequency
# Given a list of tokens and a vocabulary, compute the term frequency for each term in the vocabulary.
def compute_tf(tokens, vocab, normalize=True):
    count = Counter(tokens)
    if normalize:
        total_terms = len(tokens) or 1
        return { term: count[term] / total_terms for term in vocab }
    else:
        return { term: count[term] for term in vocab }