
import numpy as np
import joblib

from lang import utils, ranking

# ------------------------------------------------------

class PersonSentenceData:

    def __init__(self, language, param) -> None:
        
        self.word_seen_freq_dict = {}
        
        for word, _ in language.word_freq_dict.items():
            self.word_seen_freq_dict[word] = 0
        
        self.param = param
            
        # store to prevent unecessary recomputation
        self.word_freq_scores = None
        self.sentence_length_scores = None
        self.sentence_structure_scores = None
        
        self.selected = []
        
        # 1 : not seen, 0 : seen
        self.sentence_seen = np.ones(shape=(len(language.sentences)))
        
        self.ordered_sentences_indices = None
        self.last_index = None
    
# ------------------------------------------------------

class PersonWordData:

    def __init__(self, language, param) -> None:
        
        self.letter_seen_freq_dict = {}
        
        for letter, _ in language.letter_freq_dict.items():
            if letter in language.letters_in_corpus:
                self.letter_seen_freq_dict[letter] = 0
        
        self.param = param
            
        # store to prevent unecessary recomputation
        self.letter_freq_scores = None
        
        self.selected = []
        
        # 1 : not seen, 0 : seen
        self.word_seen = np.array([1 for w in language.words if w !=''])
        self.word_seen_letters = np.array([1 for w in language.words_for_letters if w !=''])
            
# ------------------------------------------------------

class Person:

    def __init__(self, language) -> None:
        self.language = language
        self.sentence_data = None
        self.word_data = None
            
    # ------------------------------------------------------
        
    def update_sentence_data(self, index, sentence, translation):
                
        self.sentence_data.sentence_seen[index] = 0

        words = utils.clean_up_sentence(sentence)
        for word in words:
            if word == '': continue
            self.sentence_data.word_seen_freq_dict[word] += 1

        self.sentence_data.selected.append({"sentence":sentence, "translations":translation, "explanation":None, "index":index})

    # ------------------------------------------------------
    
    def update_word_data(self, index, word, translation):
        
        if self.word_data.word_seen_letters[index] == 0:
            return
        
        self.word_data.word_seen_letters[index] = 0

        for letter in word:
            if letter in self.language.letters: 
                self.word_data.letter_seen_freq_dict[letter] += 1

        self.word_data.selected.append({"word":word, "translations": translation, "explanation":None, "index":index})

   # ------------------------------------------------------
    
    def choose_next(self, type):
        
        if type == "words":
            return self.choose_word()
        else:
            return self.choose_sentence()
    
    # ------------------------------------------------------
    
    def choose_word(self):
        
        if "random" in self.word_data.param.keys():
            index = np.random.randint(0, len(self.language.words))
        else:
            rt_eps = self.word_data.param['repetition_threshold']
            wl_eps = self.word_data.param['word_length_eps']
            ls_eps = self.word_data.param['letter_scores_eps']
            
            # scores based on letters seen frequencies
            letter_seen_scores_dict = ranking.item_seen_scores(self.word_data.letter_seen_freq_dict, rt_eps)

            word_letter_scores = ranking.word_letter_scores(
                self.language.words_for_letters,   
                self.language.letter_freq_scores_dict, 
                letter_seen_scores_dict, self.language.letters)
            
            # -------------------------------------------
            
            # word_length_scores
            len_scores = self.language.word_length_scores * wl_eps
            word_letter_scores *= ls_eps
            
            # compute word scores
            word_scores = (word_letter_scores + len_scores) * self.word_data.word_seen_letters
            
            # -------------------------------------------

            index = np.argmax(word_scores)
            
        word = self.language.words_for_letters[index]
        translation = self.language.words_for_letters_translations[index]
        
        self.update_word_data(index, word, translation) 

        return word, translation, index
       
    # ------------------------------------------------------

    def choose_sentence(self):
        
        sentences = self.language.sentences 
        
        if "random" in self.sentence_data.param.keys():
            index = np.random.randint(0, len(self.language.sentences))
        else:
            
            sl_eps = self.sentence_data.param['sentence_lengths_eps']
            rt_eps = self.sentence_data.param['repetition_threshold']
            rating_type = self.sentence_data.param['compute_rating']

            # scores based on word seen frequencies
            word_seen_scores_dict = ranking.item_seen_scores(self.sentence_data.word_seen_freq_dict, rt_eps)

            sentence_words_scores = ranking.sentence_words_scores(
                sentences,
                self.language.word_freq_scores_dict, 
                word_seen_scores_dict)
            
            len_scores = self.language.sentence_length_scores * sl_eps
            
            # compute sentence scores
            if rating_type == "addition":
                sentence_scores = (sentence_words_scores + len_scores) * self.sentence_data.sentence_seen
            else:
                sentence_scores = (sentence_words_scores * len_scores) * self.sentence_data.sentence_seen
            
            index = np.argmax(sentence_scores)
            
        sentence = self.language.sentences[index]
        translation = self.language.translations[index]['translations']
        
        self.update_sentence_data(index, sentence, translation) 

        return sentence, translation, index
                 
    # ------------------------------------------------------
    
    def choose_sentence_2(self):
        
        sentences = self.language.sentences 
        
        if self.sentence_data.ordered_sentences_indices is None:
                
            sl_eps = self.sentence_data.param['sentence_lengths_eps']
            rt_eps = self.sentence_data.param['repetition_threshold']
            rating_type = self.sentence_data.param['compute_rating']

            # scores based on word seen frequencies
            word_seen_scores_dict = ranking.item_seen_scores(self.sentence_data.word_seen_freq_dict, rt_eps)

            sentence_words_scores = ranking.sentence_words_scores(
                sentences,
                self.language.word_freq_scores_dict, 
                word_seen_scores_dict)
            
            len_scores = self.language.sentence_length_scores * sl_eps
            
            # compute sentence scores
            if rating_type == "addition":
                sentence_scores = (sentence_words_scores + len_scores) * self.sentence_data.sentence_seen
            else:
                sentence_scores = (sentence_words_scores * len_scores) * self.sentence_data.sentence_seen
            
            indices = np.argsort(sentence_scores)
            indices = indices[::-1]
            
            # print(sentence_scores[indices[:10]])
            
            self.sentence_data.ordered_sentences_indices = indices
            self.sentence_data.last_index = -1
            
            
        if self.sentence_data.last_index+1 < len(sentences):
            
            self.sentence_data.last_index += 1
            
            sentence_index = self.sentence_data.ordered_sentences_indices[self.sentence_data.last_index]
            sentence = sentences[sentence_index]
            translation = self.language.translations[sentence_index]['translations']

            self.update_sentence_data(sentence_index, sentence, translation) 

            return sentence, translation, sentence_index
 
        print(">> No more sentences to select")
                   
        return None

    # ------------------------------------------------------

    def create_data(self, type, language, param):
        
        if type == "words":
            self.word_data = PersonWordData(language, param)
        else:
            self.sentence_data = PersonSentenceData(language, param)
            
    # ------------------------------------------------------

    def load_data(self, path, type):
        
        if type == "words":
            self.word_data = joblib.load(path)
        else:
            self.sentence_data = joblib.load(path)
            
    # ------------------------------------------------------

    def get_data(self, type):
        
        if type == "words":
            return self.word_data
        else:
            return self.sentence_data

    # ------------------------------------------------------

# ------------------------------------------------------
