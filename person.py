
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from collections import Counter
from learning import utils
from learning import ranking

# ------------------------------------------------------

class Person:


    def __init__(self, language, repetition_threshold=5) -> None:
        
        self.language = language
    
        self.words_seen_freq = {}
        
        self.repetition_threshold = repetition_threshold
    
        for word, _ in self.language.word_freq:
            self.words_seen_freq[word] = 0
            
        # store to prevent unecessary recomputation
        self.word_freq_scores = None
        self.sentence_length_scores = None
        self.sentence_structure_scores = None
        
        self.sentences_seen_indices = []
                
    # ------------------------------------------------------
        
    def update_inner_state(self, index, sentence):
                
        self.sentences_seen_indices.append(index)

        words = utils.clean_up_sentence(sentence)
        for word in words:
            if word == '':
                continue
            
            self.words_seen_freq[word] += 1

    # ------------------------------------------------------
        
    def choose_sentence(self, word_freq_eps=1.0, word_seen_freq_eps=1.0,
                        sentence_lengths_eps=1.0, sentence_structure_freq_eps=1.0,
                        sentence_random_eps=0.0):
        
        # sentence scores -- lengths
        if self.sentence_length_scores is None:
            self.sentence_length_scores = np.array(ranking.sentence_lengths_scores(self.language.sentences))

        # -------------------------------------------
        # sentence scores -- words
        
        # scores based on word frequencies
        if self.word_freq_scores is None:
            self.word_freq_scores = ranking.freq_to_scores(self.language.word_freq)

        # scores based on word seen frequencies
        word_seen_scores = ranking.words_seen_scores(self.words_seen_freq, self.repetition_threshold)

        sentence_words_scores = ranking.sentence_words_scores(
            self.language.sentences, 
            word_freq_eps, self.word_freq_scores, 
            word_seen_freq_eps, word_seen_scores)
        
        # -------------------------------------------
        # sentence scores -- structure
            # >>> does this also affect the choice of the length of a sentence?
            
        if self.sentence_structure_scores is None: 
            self.sentence_structure_scores = np.ones(shape=(len(self.language.sentences)))

        # -------------------------------------------
        # sentence scores -- a bit of randomness
        
        sentence_random_scores = np.random.rand(len(self.language.sentences))

        # -------------------------------------------
        # sentence scores -- seen (binary for anki)
        
        sentence_scores_seen = np.ones(shape=(len(self.language.sentences)))
        
        for idx in self.sentences_seen_indices:
            sentence_scores_seen[idx] = 0
            
        # -------------------------------------------
        
        sentence_scores = sentence_words_scores + \
                            self.sentence_length_scores*sentence_lengths_eps + \
                            self.sentence_structure_scores*sentence_structure_freq_eps + \
                            sentence_random_scores * sentence_random_eps 
        
        # mask out the sentences already seen
        sentence_scores *= sentence_scores_seen

        # sort by score
        indices = np.argsort(sentence_scores)

        index = indices[-1]
        
        sentence = self.language.sentences[index]
        translation = self.language.translations[index]
        
        # -------------------------------------------

        self.update_inner_state(index, sentence)

        return sentence, translation
        
# ------------------------------------------------------


        