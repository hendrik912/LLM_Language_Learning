import numpy as np

from collections import Counter
from learning import utils

# ------------------------------------------------------

def rank_sentences(word_freq, word_seen_freq, sentence_structure_freq):
    pass

# ------------------------------------------------------

def freq_to_scores(freqs):
    
    sorted_freqs = np.array(freqs)
    freqs = [int(word[1]) for word in sorted_freqs]
    scores = [freq/max(freqs) for freq in freqs]

    score_map = {}

    for (word, _), prob in zip(sorted_freqs, scores):
        score_map[word] = prob
        
    return score_map
        
# ------------------------------------------------------

def words_seen_scores(words_seen_freq_dict, repetition_threshold):
    
    word_seen_inv = {}
    
    for word, value in words_seen_freq_dict.items():
        count = np.max([0, repetition_threshold-value])
        word_seen_inv[word] = count

    word_seen_inv = list(word_seen_inv.items())
    
    return freq_to_scores(word_seen_inv)

# ------------------------------------------------------

def sentence_lengths_scores(sentences):
    
    lengths = utils.get_sentence_lengths(sentences)

    counter = Counter(lengths)

    scores = []

    for l in lengths:
        freq = counter[l]
        scores.append(freq)
        
    scores = np.array(scores, dtype=np.float32)
    scores /= np.max(scores)
    
    return scores

# ------------------------------------------------------

def sentence_structure_scores(sentences):
    pass

# ------------------------------------------------------

def sentence_words_scores(sentences, word_freq_eps, word_freq_scores, word_seen_freq_eps, word_seen_scores):

    sentence_scores = []
                
    for sentence in sentences:
        
        words = utils.clean_up_sentence(sentence)
        words = set(words)
        
        rating = 0
        
        for word in words:
            if word == '':
                continue
            
            word_freq_rating = word_freq_eps*word_freq_scores[word]
            word_seen_rating = word_seen_freq_eps*word_seen_scores[word]
            rating += word_freq_rating * word_seen_rating
            
        rating /= len(words)
        sentence_scores.append(rating)

    sentence_scores = np.array(sentence_scores)
    sentence_scores /= np.max(sentence_scores)

    return sentence_scores

# ------------------------------------------------------
