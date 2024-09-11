import nltk
import pandas as pd
import numpy as np
import string

from tqdm.notebook import tqdm

# ------------------------------------------------------

def load_corpus(language):
    
    if language == "hindi":
        fn = "hindi.tsv"        
    elif language == "farsi":
        fn = "farsi.tsv"        
    elif language == "german":
        fn = "german.tsv"        
    else:
        print("Language not supportet")
        return []
        
    df = pd.read_csv(fn, sep='\t')
    sentences_from_df = np.array(df)

    sentences = []
    translations = []

    for sentence in sentences_from_df:
        sentences.append(sentence[1])
        translations.append(sentence[3])
        
    return sentences, translations

# ------------------------------------------------------

def clean_up_sentence(sentence):
    
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words]

    # Remove punctuation
    punctuation_table = str.maketrans('', '', string.punctuation + '।' + '।' + ':' + '|'+ ' ')
    words = [word.translate(punctuation_table) for word in words]
    
    return words

# ------------------------------------------------------

def get_sentence_lengths(sentences):
    lengths = []

    for sentence in sentences:
        words = clean_up_sentence(sentence)
        lengths.append(len(words))
    
    return lengths    

# ------------------------------------------------------

def word_frequency(sentences):
    
    word_count_dict = {}

    for sentence in tqdm(sentences):
        
        words = clean_up_sentence(sentence)

        for word in words:
            
            if word == '':
                continue
            
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
                
    # Sort words by frequency in descending order
    sorted_words = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words

# ------------------------------------------------------

def sentence_structure_frequency(sentences):
    return None

# ------------------------------------------------------
