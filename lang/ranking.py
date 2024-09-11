import numpy as np

from collections import Counter
from lang import utils

# ------------------------------------------------------

def normalize_freqs(item_freq_dict):
    max_val = max(item_freq_dict.values(), default=0)
    
    if max_val == 0:
        return {k: 0.0 for k in item_freq_dict}
    
    normed = {k: float(v) / max_val for k, v in item_freq_dict.items()}
    
    return normed

# ------------------------------------------------------

def item_seen_scores(item_seen_freq_dict, repetition_threshold):
    
    item_seen_inv = {}
    
    for word, value in item_seen_freq_dict.items():
        count = np.max([0, repetition_threshold-value])
        item_seen_inv[word] = count

    return normalize_freqs(item_seen_inv)


# def item_seen_scores_vec(item_seen_freq_dict, repetition_threshold):
    
#     item_seen_inv = []
    
#     for word, value in item_seen_freq_dict.items():
#         count = np.max([0, repetition_threshold-value])
#         item_seen_inv[word] = count

#     return normalize_freqs(item_seen_inv)

# ------------------------------------------------------

def lengths_scores(list_, type):
    
    if type == "words":
        lengths = [len(word) for word in list_]
    else: # sentence
        lengths = utils.get_sentence_lengths(list_)

    counter = Counter(lengths)

    scores = []

    for l in lengths:
        freq = counter[l]
        scores.append(freq)
        
    scores = np.array(scores, dtype=np.float32)
    scores /= np.max(scores)
    
    return scores

# ------------------------------------------------------

def sentence_words_scores(sentences, word_freq_scores_dict, word_seen_scores_dict):

    sentence_scores = []
                
    for sentence in sentences:
        
        words = utils.clean_up_sentence(sentence)
        words = set(words)
        
        rating = 0
        
        for word in words:
            if word == '':
                continue
            
            try:
                word_freq_rating = word_freq_scores_dict[word]
                word_seen_rating = word_seen_scores_dict[word]
                rating += word_freq_rating * word_seen_rating
            except:
                print(word)
                
                
        rating /= len(words)
        sentence_scores.append(rating)

    sentence_scores = np.array(sentence_scores)
    sentence_scores /= np.max(sentence_scores)

    return sentence_scores


# ------------------------------------------------------

def word_letter_scores(words, letter_freq_scores_dict, letter_seen_scores_dict, alphabet):

    word_scores = []
                
    for word in words:
        
        if len(word) == 0:
            print(">>", word)
            continue
        
        letters = [c for c in word]

        letters = set(letters)
        
        rating = 0
        
        for letter in letters:
            
            if letter not in alphabet:
                continue
            
            letter_freq_rating = letter_freq_scores_dict[letter]
            letter_seen_rating = letter_seen_scores_dict[letter]
            rating += letter_freq_rating * letter_seen_rating
            
        rating /= len(letters)
        word_scores.append(rating)

    word_scores = np.array(word_scores)
    word_scores /= (np.max(word_scores) + 0.00000000001)

    return word_scores

# ------------------------------------------------------
