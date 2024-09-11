import nltk
import pandas as pd
import numpy as np
import string
import matplotlib
import matplotlib.pyplot as plt

from learning import utils
from tqdm.notebook import tqdm
from collections import Counter
from matplotlib.font_manager import FontProperties

# ------------------------------------------------------

class Language:
    
    def __init__(self, lang_str) -> None:
        
        self.sentences = None
        self.translations = None
        self.word_freq = None
        
        self.lang_str = lang_str
        self.load_language_data() 
        
    def load_language_data(self):
        
        print("Load corpus")
        self.sentences, self.translations = utils.load_corpus(self.lang_str)
        
        print("Compute frequencies")
        self.word_freq = utils.word_frequency(self.sentences)
        
        print("Compute sentence structure frequency (not implemented)")
        self.sentence_structure_freq = utils.sentence_structure_frequency(self.sentences)
        
# ------------------------------------------------------
