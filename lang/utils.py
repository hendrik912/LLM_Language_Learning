import nltk
import pandas as pd
import numpy as np
import string
import stanza
import torch
import gc
import os
import string
import unicodedata
import joblib

from huggingface_hub import login
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langdetect import detect
from llmlingua import PromptCompressor

from tqdm.notebook import tqdm


# ------------------------------------------------------

def get_language_characters(language_name):
    
    if language_name == "hindi":
        return get_devanagari_characters()
    
    elif language_name == "farsi":
        return get_farsi_characters()
    
    elif language_name == "german":
        return get_german_characters()
        
    else:
        print(f"No letters for language {language_name}")
        return []
    
# ------------------------------------------------------

def get_german_characters():
    german_chars = []
    
    # Basic Latin characters (A-Z, a-z)
    basic_latin_range = (0x0041, 0x005A)  # A-Z
    basic_latin_lower_range = (0x0061, 0x007A)  # a-z
    
    for codepoint in range(basic_latin_range[0], basic_latin_range[1] + 1):
        german_chars.append(chr(codepoint))
    
    for codepoint in range(basic_latin_lower_range[0], basic_latin_lower_range[1] + 1):
        german_chars.append(chr(codepoint))
    
    # German special characters (umlauts and ß)
    german_special_chars = [
        '\u00C4',  # Ä
        '\u00E4',  # ä
        '\u00D6',  # Ö
        '\u00F6',  # ö
        '\u00DC',  # Ü
        '\u00FC',  # ü
        '\u00DF',  # ß
    ]
    
    german_chars.extend(german_special_chars)
    
    return german_chars

# ------------------------------------------------------

def get_devanagari_characters():
    devanagari_chars = []
    
    # Devanagari script range in Unicode
    devanagari_range = (0x0900, 0x097F)
    
    for codepoint in range(devanagari_range[0], devanagari_range[1] + 1):
        character = chr(codepoint)
        category = unicodedata.category(character)
        
        if category.startswith('Mc') or category.startswith('Mn'):
            # 'Mc' and 'Mn' categories represent combining characters (diacritics)
            devanagari_chars.append(character)
        elif category == 'Lo' or category == 'Nd':
            # 'Lo' category represents letter characters
            # 'Nd' category represents decimal digit characters
            devanagari_chars.append(character)
    
    # Adding Devanagari numerals (0-9)
    devanagari_numerals = [chr(0x0966 + i) for i in range(10)]
    devanagari_chars.extend(devanagari_numerals)
    
    return devanagari_chars
        
# ------------------------------------------------------

def get_farsi_characters():
    farsi_chars = []
    
    # Arabic script range in Unicode (including Persian characters)
    arabic_range = (0x0600, 0x06FF)
    arabic_supplement_range = (0x0750, 0x077F)
    arabic_extended_a_range = (0x08A0, 0x08FF)
    
    # Iterate over the ranges and collect characters
    for codepoint in range(arabic_range[0], arabic_range[1] + 1):
        character = chr(codepoint)
        category = unicodedata.category(character)
        
        if category.startswith('Mc') or category.startswith('Mn'):
            # 'Mc' and 'Mn' categories represent combining characters (diacritics)
            farsi_chars.append(character)
        elif category == 'Lo' or category == 'Nd':
            # 'Lo' category represents letter characters
            # 'Nd' category represents decimal digit characters
            farsi_chars.append(character)
    
    for codepoint in range(arabic_supplement_range[0], arabic_supplement_range[1] + 1):
        character = chr(codepoint)
        category = unicodedata.category(character)
        
        if category.startswith('Mc') or category.startswith('Mn'):
            farsi_chars.append(character)
        elif category == 'Lo' or category == 'Nd':
            farsi_chars.append(character)
    
    for codepoint in range(arabic_extended_a_range[0], arabic_extended_a_range[1] + 1):
        character = chr(codepoint)
        category = unicodedata.category(character)
        
        if category.startswith('Mc') or category.startswith('Mn'):
            farsi_chars.append(character)
        elif category == 'Lo' or category == 'Nd':
            farsi_chars.append(character)

    # Additional Persian-specific characters not included in the main Arabic ranges
    persian_specific_chars = [
        '\u067E', # Pe (پ)
        '\u0686', # Che (چ)
        '\u0698', # Ze (ژ)
        '\u06AF', # Gaf (گ)
    ]
    
    farsi_chars.extend(persian_specific_chars)
    
    return farsi_chars

# ------------------------------------------------------

def load_corpus(base_dir, language):
    
    fn = os.path.join(base_dir, language + ".tsv")        
    
    df = pd.read_csv(fn, sep='\t')
    sentences_from_df = np.array(df)

    sentence_translation_dict = {}

    for sentence in sentences_from_df:
        try:
            sentence_translation_dict[sentence[1]]['translations'].append(sentence[3])
        except:
            sentence_translation_dict[sentence[1]] = {}
            sentence_translation_dict[sentence[1]]['translations'] = [sentence[3]]        
            sentence_translation_dict[sentence[1]]['explanation'] = None        

    sentences = []
    translations = []

    for sent, trans in sentence_translation_dict.items():
        sentences.append(sent)
        translations.append(trans)
        
    return sentences, translations

# ------------------------------------------------------

def clean_up_sentence(sentence):
    
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words]

    # Remove punctuation
    punctuation_table = str.maketrans('', '', string.punctuation + '।' + '।' + ':' + '|'+ ' ')
    words = [word.translate(punctuation_table) for word in words]
    
    return words


# def clean_up_sentence(sentence):
#     # Efficient split and lowercase conversion
#     words = (word.lower() for word in sentence.split())
    
#     # Define the translation table for removing punctuation
#     punctuation_table = str.maketrans('', '', string.punctuation + '।' + '।' + ':' + '|'+ ' ')
    
#     # Remove punctuation and filter out empty words
#     words = [word.translate(punctuation_table) for word in words if word.translate(punctuation_table)]
    
#     return words

# ------------------------------------------------------

def get_sentence_lengths(sentences):
    lengths = []

    for sentence in sentences:
        words = clean_up_sentence(sentence)
        lengths.append(len(words))
    
    return lengths    

# ------------------------------------------------------

def word_frequency(sentences):
    
    word_freq_dict = {}

    for sentence in tqdm(sentences):
        words = clean_up_sentence(sentence)

        for word in words:
            if word == '': 
                continue
            elif word in word_freq_dict:
                word_freq_dict[word] += 1
            else:
                word_freq_dict[word] = 1
    
    return word_freq_dict

# ------------------------------------------------------

def sentence_structure_frequency(sentence_structure_count_dict, sentences):
    return None

# ------------------------------------------------------

def compute_sentence_structure_count_dict(sentences, lang="hi"):
    
    nlp = stanza.Pipeline(lang)

    structure_dict = {}

    for sentence in tqdm(sentences):
        
        doc = nlp(sentence)

        for sent in doc.sentences:
            
            structure = ""
            for word in sent.words:
                structure += word.pos + "+" 
                
            structure = structure[:-1]

            try:
                structure_dict[structure] += 1
            except:
                structure_dict[structure] = 1
                
    return structure_dict

# ------------------------------------------------------

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# -----------------------------------------------------------------------

def load_LLM(token=""):
    
    login(token=token)
    # transformers.logging.set_verbosity_error()

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_name) # , padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        return_full_text=False,
    )

    return pipe

# -------------------------------------------------------------------------

def prompt_LLM(LLM, prompt, max_new_tokens=500):

    flush()

    with torch.no_grad():

        outputs = LLM(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            num_return_sequences=1,
            temperature=0.25, top_k=50, top_p=0.95,
        )

    output = outputs[0]["generated_text"]
    output = output.replace('"', "'")
    
    return output

# -----------------------------------------------------------------------

def compress_prompt(prompt):
    
    flush()

    # Or use LLMLingua-2-small model
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
    )

    compressed_prompt = llm_lingua.compress_prompt(prompt) 
    
    del llm_lingua
    flush()

    return compressed_prompt

# -----------------------------------------------------------------------

def dict_to_str(dict_):
    output = ""
    for k,v in dict_.items():
        v_str = str(v)
        v_str = v_str.replace(".",",")
        output += f"{k}_{v_str}_"
    return output[:-1]

# -----------------------------------------------------------------------

def rolling_mean(numbers, k):
    """Compute the rolling mean of a list of numbers with window size k."""
    return [np.mean(numbers[i:i+k]) for i in range(len(numbers)-k+1)]

def load_LLM(token=""):

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_name) # , padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        return_full_text=False,
    )

    return pipe

# -------------------------------------------------------------------------

def extract_data(params, language, result_path, lang_str, type="sentences"):
        
    lists_of_sum_items_per_iter = []
    lists_of_new_items_per_sentence = []
    lists_of_sum_covered_freq_per_iter = []
    lists_of_covered_freq_per_iter = []

    legend_labels = []

    if type in ['sentences', 'sentence']:
        item_freq_dict = language.word_freq_dict 
    else:
        item_freq_dict = language.letter_freq_dict
        
    for param in params:
        
        legend_labels.append(dict_to_str(param))
        
        base_fn = f"{lang_str}_{type[:-1]}_data"
        
        path = os.path.join(result_path, base_fn + "_" + dict_to_str(param))
        
        person_data = joblib.load(path)
        
        sum_items_per_iter = []
        new_items_per_sentence = []
        sum_covered_freq_per_iter = []
        covered_freq_per_iter = []
        
        all_items = set()
        total_freq_covered = 0

        item_key = type[:-1]
        
        for idx, entry_dict  in enumerate(person_data.selected):

            item = entry_dict[item_key]

            if type in ['sentences', 'sentence']:
                items = clean_up_sentence(item)
            else:
                items = [c for c in item]

            num_new = 0
            new_freq_covered = 0

            for word in items:
                if word not in all_items: 
                    if word == '' or (type == "words" and word not in language.letters):
                        continue
                        
                    num_new += 1
                    new_freq_covered += item_freq_dict[word]
                    
                all_items.add(word)

            covered_freq_per_iter.append(new_freq_covered)
            total_freq_covered += new_freq_covered
            sum_covered_freq_per_iter.append(total_freq_covered)

            sum_items_per_iter.append(len(all_items))
            new_items_per_sentence.append(num_new)

        # ----------------
        
        lists_of_sum_items_per_iter.append(sum_items_per_iter)
        lists_of_new_items_per_sentence.append(new_items_per_sentence)
        lists_of_sum_covered_freq_per_iter.append(sum_covered_freq_per_iter)
        lists_of_covered_freq_per_iter.append(covered_freq_per_iter)

    return lists_of_sum_items_per_iter, lists_of_new_items_per_sentence, lists_of_sum_covered_freq_per_iter, lists_of_covered_freq_per_iter, legend_labels

# -------------------------------------------------------------------------

def generate_explanation(person_data, person_save_path, LLM, prompt_template, task, examples, language_path, language, gen_type, 
                         recompute_if_exists=False, stop_after_n=None, num_tokens_per_word=150, print_output=False):
    
    updated_language = False
    updated_person_word_data = False

    idx = 0
    for item_dict in tqdm(person_data.selected):
        if print_output:
            print(f"{idx+1}/{len(person_data.selected)}")
        
        if stop_after_n is not None and idx >= stop_after_n: 
            print("Stop execution due to 'stop_after_n")
            break

        index = item_dict['index']
        input = item_dict[gen_type]
                
        translation = ""
        
        for t in item_dict['translations']:
            translation += t

        if item_dict['explanation'] is not None and not recompute_if_exists:
            if print_output:
                print("Explanation in sentence_data")
            
            if not language.get_explanation(index, gen_type):
                if print_output:
                    print("Add explanation to language-object")
                language.add_explanation(item_dict['explanation'], index, gen_type)
                updated_language = True
        
        elif language.get_explanation(index, gen_type) is not None and not recompute_if_exists:
            if print_output:
                print("Explanation in language object")
            
            answer = language.get_explanation(index, type)
            person_data.selected[idx]['explanation'] = answer
            updated_person_word_data = True
            
        else:         
            prompt = prompt_template.substitute(task=task, examples=examples, input=input, translation=translation)
            
            if gen_type == "sentence":
                num_words = len(clean_up_sentence(input))
            else:
                num_words = 1 # len(input) // 2
            
            if print_output:
                print(">> Tokens", num_tokens_per_word*num_words)
            
            answer = prompt_LLM(LLM, prompt, max_new_tokens=num_tokens_per_word*num_words)
        
            person_data.selected[idx]['explanation'] = answer
            updated_person_word_data = True
        
            language.add_explanation(item_dict['explanation'], index, gen_type)
            
            updated_language = True    
            
            if idx < 10 and print_output:
                print("_"*10)
                print("Input:", input)
                print("Explanation:")
                print(answer)

        if idx % 5 == 0:
            if updated_person_word_data:
                # print("Save person word data")
                joblib.dump(person_data, person_save_path)
                updated_person_word_data = False
            
            if updated_language:
                # print("Save language object")
                joblib.dump(language, language_path)
                updated_language = False
    
        idx += 1
        
    joblib.dump(person_data, person_save_path)
    joblib.dump(language, language_path)
            
# -------------------------------------------------------------------------

def delete_mp3_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out only .mp3 files
    mp3_files = [file for file in files if file.endswith('.mp3')]
    
    # Delete each .mp3 file
    for mp3_file in mp3_files:
        file_path = os.path.join(directory, mp3_file)
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
            
# -------------------------------------------------------------------------
