import os
import numpy as np
import random

from tqdm.notebook import tqdm
from lang import utils, ranking

# ------------------------------------------------------

class Language:
    
    def __init__(self, base_dir, lang_str, nouns=None) -> None:
        
        self.base_dir = base_dir
        self.lang_str = lang_str

        self.letters = None
        self.words = None
        self.sentences = None
        
        self.translations = None
        
        self.word_freq_dict = None
        self.word_freq_scores_dict = None
        self.word_freq_scores = None
        self.sentence_length_scores = None
        
        self.letter_freq_dict = None
        self.letter_freq_scores_dict = None
        self.letter_freq_scores = None
        self.word_length_scores = None
        
        self.letters_in_corpus = None
        
        self.sentence_explanations = None
        self.word_explanations = None

        self.nouns = nouns if nouns is not None else []

        # --------------------------------------------
                
        self.load_data() 
        self.init_sentence_data()
        self.init_letter_data()
        
    # ------------------------------------------------------

    def load_data(self):
        print("load data")
        
        self.sentences, self.translations = utils.load_corpus(self.base_dir, self.lang_str)
        
        if len(self.sentences) > 15000:
            print(">> Limit sentences to 15000")

            permutation = np.random.permutation(len(np.array(self.sentences)))

            self.sentences = list(np.array(self.sentences)[permutation])
            self.translations = list(np.array(self.translations)[permutation])
        
            self.sentences = self.sentences[:15000]
        else:
            print(len(self.sentences))
        
        
        # indices = [i for i,s in enumerate(self.sentences) if len(utils.clean_up_sentence(s)) == 1]
        # self.words_for_letters = [self.sentences[i] for i in indices]
        # self.words_for_letters_translations = [self.translations[i] for i in indices]

        self.words_for_letters = [n for n,t in self.nouns]
        self.words_for_letters_translations = [t for n,t in self.nouns]
        
        self.letters = utils.get_language_characters(self.lang_str)

        self.words = set()
        
        for sentence in tqdm(self.sentences):
            words = utils.clean_up_sentence(sentence)
            for word in words:
                if word != '': 
                    self.words.add(word)
                
        self.words = list(self.words)

    # ------------------------------------------------------
        
    def init_sentence_data(self):
        print("init sentence data")
        
        self.word_freq_dict = utils.word_frequency(self.sentences)    
        
        self.word_freq_scores_dict = ranking.normalize_freqs(self.word_freq_dict)
        
        self.word_freq_scores = np.array([v for _,v in self.word_freq_scores_dict.items()], dtype=np.float32)
        
        self.sentence_length_scores = np.array(ranking.lengths_scores(self.sentences, type="sentence"))
        
        self.sentence_explanations = []
        for _ in range(len(self.sentences)):
            self.sentence_explanations.append(None) 
        
    # ------------------------------------------------------

    def init_letter_data(self):
        print("init letter data")
                
        self.letters_in_corpus = set()
        
        
        for word in self.words_for_letters:
            for c in word:
                self.letters_in_corpus.add(c)
                
        try:
            self.word_length_scores = np.array(ranking.lengths_scores(self.words_for_letters, type="words"))
        except:
            self.word_length_scores = None
            
        # letter frequencies
        self.letter_freq_dict = {}
        
        for letter in self.letters:
            self.letter_freq_dict[letter] = 0

        for sentence in tqdm(self.sentences):
            words = utils.clean_up_sentence(sentence)

            for word in words:
                for letter in word:
                    if letter in self.letters:
                        self.letter_freq_dict[letter] += 1

        self.letter_freq_scores = np.array([v for (_,v) in self.letter_freq_dict.items()], dtype=np.float32)
             
        max_val = max(self.letter_freq_scores)
        
        self.letter_freq_scores /= max_val
        
        self.letter_freq_scores_dict = {c:v/max_val for (c,v) in self.letter_freq_dict.items()}
        
        self.word_explanations = []
        for _ in range(len(self.words_for_letters)):
            self.word_explanations.append(None) 
            
    # ------------------------------------------------------

    def add_explanation(self, explanation, index, type):
        if type == "word":
            self.word_explanations[index] = explanation
        else:
            self.sentence_explanations[index] = explanation
        
    # ------------------------------------------------------

    def get_explanation(self, index, type):
        
        if type == "word":
            return self.word_explanations[index]
        else:
            return self.sentence_explanations[index]
        
# ------------------------------------------------------
