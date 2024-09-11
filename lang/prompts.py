

def get_examples_for_prompt(type, lang_str, detailed=False):
    key = "detailed" if detailed else "minimal"
    assert example_dict[key][type][lang_str] is not None
    return example_dict[key][type][lang_str]


hindi_examples_words = """

Word: लैटिन
Translation: Latin

\\
    
The Hindi word "लैटिन" translates to "Latin" in English. Let's go over each letter in detail, including the sounds and how they combine to form the word.

1. ल (la)
Letter: ल
Sound: The sound is similar to the 'l' in "lamp."
Description: This is a straightforward consonant letter, representing the sound 'la.'
2. ै (ai)
Diacritic: ै (ai)
Sound: The sound is like the 'ai' in "air."
Description: This is a vowel diacritic called "ai-matra." It modifies the preceding consonant (ल) to change the vowel sound from the inherent 'a' to 'ai.' When combined with ल, it changes ल (la) to लै (lai).
3. ट (ṭa)
Letter: ट
Sound: The sound is similar to the 't' in "table," but it is retroflex. The tongue curls back to the roof of the mouth, producing a hard 'ṭa' sound.
Description: This is a retroflex consonant, representing the sound 'ṭa.'
4. ि (i)
Diacritic: ि (i)
Sound: The sound is like the 'i' in "bit."
Description: This is a vowel diacritic called "i-matra." It modifies the preceding consonant (ट) to change the vowel sound from the inherent 'a' to 'i.' When combined with ट, it changes ट (ṭa) to टि (ṭi). Note that the diacritic is placed before the consonant in writing but pronounced after it.
5. न (na)
Letter: न
Sound: The sound is similar to the 'n' in "name."
Description: This is a straightforward consonant letter, representing the sound 'na.'
Putting it all together: लैटिन (Laiṭina)
लै (lai): The combination of ल (la) and ै (ai) produces the syllable 'lai.'
टि (ṭi): The combination of ट (ṭa) and ि (i) produces the syllable 'ṭi.' Note the position of the diacritic before ट but it follows the consonant sound.
न (na): This is a standalone consonant 'na.'
When you combine these syllables, you get लैटिन (Laiṭina), which phonetically approximates the English word "Latin."

Summary:
लैटिन is composed of:
ल (la) + ै (ai) = लै (lai)
ट (ṭa) + ि (i) = टि (ṭi)
न (na) = न (na)
Together, लैटिन reads as "Laiṭina," closely matching the pronunciation of "Latin."
Each letter and diacritic in "लैटिन" contributes to the overall pronunciation, transforming simple consonants and vowels into the composite sounds needed to approximate the English word "Latin" in Hindi.
"""

hindi_examples_sentences = """

Example 1:

Sentence: अच्छा वाला दे दीजिए?
Translation: Can you give me a good one?

\\ 

The given Hindi sentence 'अच्छा वाला दे दीजिए' can be broken down as follows:

1. 'अच्छा' (acchaa) is an adjective that modifies the expected object, indicating that the expected object should be of good quality.
2. 'वाला' (vaalaa) is a numeral indicating that the expected object is singular.
3. 'दे' (de) is the imperative form of the verb 'देना' (to give), indicating that the speaker is giving an instruction to someone to perform the action of giving.
4. 'दीजिए' (deeje) is the plural imperative form of the verb 'दे' (to give), indicating that the speaker is instructing multiple people to perform the action of giving.


Example 2:

Sentence: मैं एक किताब पढ़ रहा था
Translation: I was reading a book.

\\

The given Hindi sentence 'मैं एक किताब पढ़ रहा था' can be broken down as follows:

1. 'मैं' (main) is a pronoun meaning 'I,' indicating the subject of the sentence.
2. 'एक' (ek) is an indefinite article meaning 'a' or 'one,' used to specify a single book.
3. 'किताब' (kitaab) is a noun meaning 'book.'
4. 'पढ़' (padh) is the root form of the verb 'पढ़ना' (padhna - to read), indicating the action of reading.
5. 'रहा' (raha) is the present participle form of the verb 'पढ़ना' (padhna - to read), used to indicate the continuous aspect of the verb. This form is masculine singular.
6. 'था' (tha) is the past tense form of the verb 'होना' (hona - to be), which completes the past continuous tense for a masculine subject.


Example 3:

Sentence: मैंने सेब खाया
Translation: I ate an apple.

\\

The given Hindi sentence 'मैंने सेब खाया' can be broken down as follows:

1. 'मैंने' (maine) is a pronoun meaning 'I,' but in the perfective aspect. It's derived from 'मैं' (main - I) and is used to indicate that the action was completed by the subject.
2. 'सेब' (seb) is a noun meaning 'apple.'
3. 'खाया' (khaya) is the past participle form of the verb 'खाना' (khaana - to eat), indicating that the action of eating was completed. The form 'खाया' (khaya) is masculine singular.

"""


hindi_examples_sentences_detailed = """
Example 1:

Sentence: तुम आज क्यों नहीं आये?
Translation: Why didn't you come today?

\\
  
"तुम" (tum)

Translation: you
Grammatical Aspect: Pronoun
Case: Nominative (subject of the sentence)
Number: Singular/Plural (context-dependent)
Person: Second person
Description: "तुम" is an informal second person pronoun used to address someone directly. It serves as the subject of the sentence.

"आज" (aaj)

Translation: today
Grammatical Aspect: Adverb
Function: Indicates the time when the action is supposed to take place
Description: "आज" is a temporal adverb specifying the time frame of the action, indicating that the action in question was expected to occur today.

"क्यों" (kyon)

Translation: why
Grammatical Aspect: Interrogative adverb
Function: Used to ask for the reason or cause of an action or situation
Description: "क्यों" is used in questions to inquire about the reason or purpose behind the action being discussed, setting the sentence up as a question.

"नहीं" (nahin)

Translation: not
Grammatical Aspect: Adverb
Function: Negates the verb in the sentence
Description: "नहीं" is used to negate the verb phrase. In this sentence, it negates the action of "आये" (came), indicating that the action did not take place.

"आये" (aaye)

Translation: came
Grammatical Aspect: Verb
Tense: Simple past
Mood: Indicative
Voice: Active
Number: Plural (but often used for respect in singular)
Person: Second person
Description: "आये" is the past tense form of "आना" (to come). In this context, it indicates that the action of coming did not happen.
Full Explanation
Sentence Structure: The sentence "तुम आज क्यों नहीं आये?" follows the typical Hindi word order for questions with a question word: subject ("तुम") + time adverb ("आज") + interrogative adverb ("क्यों") + negation ("नहीं") + verb ("आये").
Interrogative Construction: "क्यों" initiates the question by asking for the reason behind the action.
Temporal Aspect: "आज" places the action in the present day.
Negation: "नहीं" negates the action, indicating that the expected action did not occur.
Verb Tense: The verb "आये" is in the simple past tense, indicating that the action of coming did not happen up to the present time.
When the sentence components come together, the structure in Hindi follows this pattern: Subject ("तुम") + Time Adverb ("आज") + Interrogative Adverb ("क्यों") + Negation ("नहीं") + Verb ("आये"). Therefore, "तुम आज क्यों नहीं आये?" asks why the addressed person did not come today.  
    
    
    
    
Example 2:

Sentence: मैंने कल किताब पढ़ी
Translation: I read a book yesterday.

\\
    
"मैंने" (maine)

Translation: I (with an auxiliary verb)
Grammatical Aspect: Pronoun with auxiliary verb construction
Case: Ergative (used with transitive verbs in perfect tenses)
Number: Singular
Person: First person
Description: "मैंने" is the first person singular pronoun "मैं" (I) in the ergative case, used with perfect tenses in Hindi. It indicates that the speaker performed the action of the verb.

"कल" (kal)

Translation: yesterday
Grammatical Aspect: Adverb
Function: Indicates the time when the action took place
Description: "कल" is a temporal adverb that can mean "yesterday" or "tomorrow" based on the context. In this sentence, it specifies the time frame of the action as "yesterday."

"किताब" (kitaab)

Translation: book
Grammatical Aspect: Noun
Case: Direct (as the object of the verb)
Gender: Feminine
Number: Singular
Description: "किताब" is a feminine noun meaning "book." In this sentence, it serves as the direct object of the verb "पढ़ी."

"पढ़ी" (paṛhī)

Translation: read
Grammatical Aspect: Verb
Tense: Simple past (perfect)
Mood: Indicative
Voice: Active
Number: Singular
Gender: Feminine (to agree with the feminine noun "किताब")
Person: First person (matching with "मैंने")
Description: "पढ़ी" is the past tense form of "पढ़ना" (to read). The verb form agrees in gender and number with the direct object "किताब," which is feminine singular.
Full Explanation
Sentence Structure: The sentence "मैंने कल किताब पढ़ी" follows the typical Hindi word order: subject ("मैंने") + time adverb ("कल") + object ("किताब") + verb ("पढ़ी").
Tense and Aspect: The verb "पढ़ी" is in the simple past (perfect) tense, indicating that the action of reading was completed in the past. The auxiliary verb construction with "मैंने" shows that the action was performed by the speaker.
Temporal Aspect: "कल" places the action in the past, specifically "yesterday."
Gender Agreement: The verb "पढ़ी" agrees in gender with the noun "किताब," both being feminine.
When the sentence components come together, the structure in Hindi follows this pattern: Subject with Auxiliary ("मैंने") + Time Adverb ("कल") + Object ("किताब") + Verb ("पढ़ी"). Therefore, "मैंने कल किताब पढ़ी" conveys that the speaker read a book yesterday.

"""

farsi_examples_words = """

Example 1:

Word: خانه
Translation: House

\\
    
The Farsi word "خانه," meaning "House," is composed of four letters: خ (Kheh), ا (Alef), ن (Noon), and ه (Heh). Kheh, pronounced /kh/ and romanized as "kh," appears as a vertical stroke with a dot above when isolated and connects to the following letter in its initial form (خـ). Alef, pronounced /a/ and romanized as "a," is a simple vertical stroke that does not connect to the next letter, maintaining its isolated form (ا). Noon, pronounced /n/ and romanized as "n," is a vertical stroke with a dot above, connecting to both sides in its medial form (ـنـ). Heh, pronounced /h/ and romanized as "h," has a loop-like shape and connects to the previous letter in its final form (ـه). In the word "خانه," Kheh connects to Alef, which stands alone and does not connect to Noon. Noon connects to Heh, which forms its final shape, resulting in the smooth and unified script "خانه," romanized as "khaneh".

Example 2:

Word: درخت
Translation: Tree

\\
    
The Farsi word "درخت," meaning "Tree," is composed of four letters: د (Dal), ر (Reh), خ (Kheh), and ت (Teh). Dal, pronounced /d/ and romanized as "d," appears as a diagonal stroke and does not connect to the following letter, maintaining its isolated form (د). Reh, pronounced /r/ and romanized as "r," is a single diagonal stroke and also does not connect to the next letter, keeping its isolated form (ر). Kheh, pronounced /kh/ and romanized as "kh," is a vertical stroke with a dot above and connects to both sides in its medial form (ـخـ). Teh, pronounced /t/ and romanized as "t," is a vertical stroke with two dots above, connecting to the previous letter in its final form (ـت). In the word "درخت," Dal stands alone and does not connect to Reh. Reh also stands alone and does not connect to Kheh. Kheh connects to Teh, which forms its final shape, resulting in the smooth and unified script "درخت," romanized as "derakht".

"""

# Word:  خانه
# Translation: House

farsi_examples_words_detailed = """

Word: خانه
Translation: House

\\
     
Detailed Analysis of Each Letter in خانه:
خ (Kheh):

Isolated Form: خ
Sound: /x/
Romanization: "kh"
Description: In its isolated form, Kheh appears as a single vertical stroke with a dot above.
Joined Form: When joined with Alef, Kheh takes the initial form (خـ). It has a tail extending to the left to connect with the next letter.
ا (Alef):

Isolated Form: ا
Sound: /ɒː/
Romanization: "a"
Description: Alef appears as a simple vertical stroke with no dots in both isolated and initial forms.
Joined Form: Alef does not change in its joined form but does not connect to the following letter (ـا).
ن (Noon):

Isolated Form: ن
Sound: /n/
Romanization: "n"
Description: In its isolated form, Noon is a single vertical stroke with a dot above.
Joined Form: When joined in the middle of a word, Noon changes to (ـنـ) with a stroke extending to the right and a dot above.
ه (Heh):

Isolated Form: ه
Sound: /h/
Romanization: "h"
Description: In its isolated form, Heh has a loop-like shape with two dots above.
Joined Form: At the end of the word, Heh changes to its final form (ـه) with a tail extending to the left.
How They Combine (Joining and Ligatures):
خانه demonstrates the cursive and connected nature of Persian script:

خ (Kheh) at the beginning of the word connects to ا (Alef). In this position, Kheh takes the form (خـ), with a connection point to the left.
ا (Alef) follows Kheh but does not connect to the next letter. Thus, it maintains its standalone form (ـا).
ن (Noon) follows Alef and is in the middle of the word, taking the form (ـنـ) with connection points on both sides.
ه (Heh) is at the end of the word and takes its final form (ـه).
Example:
When written together, the word خانه (Kheh + Alef + Noon + Heh) forms a smooth and connected script. Here's a step-by-step visualization of the transformation:

خـ (Kheh) connects to ا (Alef).
ـا (Alef) stands alone and does not connect to the following letter.
ـنـ (Noon) connects to ـه (Heh).
ـه (Heh) is at the end and forms its final shape.
So, the final word خانه (khaneh) is a seamless combination of individual letters adapting to their positions and forming a unified word in Persian script.

"""

farsi_examples_sentences = None


german_examples_sentences = """

Example 1:

Sentence: Ich ging in die Bibliothek
Translation: I went to the library.

The given German sentence 'Ich ging in die Bibliothek' can be broken down as follows:

1. 'Ich' (I) is a pronoun meaning 'I,' indicating the subject of the sentence.
2. 'ging' (went) is the past tense form of the verb 'gehen' (to go), indicating the action took place in the past.
3. 'in' (in/to) is a preposition meaning 'in' or 'to,' used here to indicate direction towards a location.
4. 'die' (the) is the definite article in the accusative case, feminine singular, corresponding to the noun 'Bibliothek.'
5. 'Bibliothek' (library) is a noun meaning 'library.'

So, the sentence structure indicates that the speaker went to a specific location, the library, in the past.

Example 2:

Sentence: Warum bist du heute nicht gekommen?
Translation: Why didn't you come today?

\\

1. Warum' (Why) is an interrogative adverb used to ask for the reason or cause of something.
2. 'bist' (are) is the second person singular form of the verb 'sein' (to be), used here as an auxiliary verb in the present perfect tense.
3. 'du' (you) is a pronoun meaning 'you,' indicating the subject of the sentence.
4. 'heute' (today) is an adverb indicating the time, specifying that the action in question relates to today.
5. 'nicht' (not) is a negation adverb, indicating that the action did not happen.
6.'gekommen' (come) is the past participle form of the verb 'kommen' (to come), used with 'bist' to form the present perfect tense.

So, the sentence structure is asking for the reason why the person did not come today, using the present perfect tense to frame the question in the context of completed action (not coming).
    
"""

german_examples_sentences_detailed = """

Example 1:

Sentence: Ich ging in die Bibliothek
Translation: I went to the library.

\\
    
"Ich"

Translation: I
Grammatical Aspect: Pronoun
Case: Nominative (subject of the sentence)
Number: Singular
Person: First person
Description: "Ich" is the nominative case pronoun used to indicate the speaker or the person performing the action in the sentence. It is used here as the subject of the verb "ging."

"ging"

Translation: went
Grammatical Aspect: Verb
Tense: Simple past (Präteritum)
Mood: Indicative
Voice: Active
Person: First person singular (ich)
Verb Type: Irregular verb (past form of "gehen")
Description: "ging" is the past tense form of "gehen" (to go). In German, the simple past tense (Präteritum) is often used in written narratives to indicate actions that happened in the past. It conveys that the action of going occurred at a specific time in the past.

"in"

Translation: in, to
Grammatical Aspect: Preposition
Case Governed: Accusative (indicating direction/movement)
Description: "in" is a preposition used to indicate direction towards a place or into an enclosed space. When used with verbs of movement, it often governs the accusative case to show the destination of the movement.

"die"

Translation: the
Grammatical Aspect: Definite article
Case: Accusative (object of the preposition "in")
Gender: Feminine
Number: Singular
Description: "die" is the definite article for feminine nouns in the nominative and accusative singular. Here, it is used in the accusative case because "Bibliothek" is the direct object of the preposition "in," indicating the destination of the action.

"Bibliothek"

Translation: library
Grammatical Aspect: Noun
Case: Accusative (object of the preposition "in")
Gender: Feminine (die Bibliothek)
Number: Singular
Description: "Bibliothek" is a feminine noun meaning "library." In this sentence, it is used in the accusative case as the object of the preposition "in," indicating the place to which the speaker went.
Full Explanation
Sentence Structure: The sentence "Ich ging in die Bibliothek" follows the typical German word order for a main clause: subject (Ich) - verb (ging) - prepositional phrase (in die Bibliothek).
Temporal Aspect: The verb "ging" is in the simple past tense, indicating that the action took place in the past. This form is often used in written narratives and reports to describe past events.
Direction and Place: The prepositional phrase "in die Bibliothek" indicates movement towards a specific place, the library. The preposition "in" governs the accusative case here because it denotes direction.
When the sentence components come together, the structure in German follows this pattern: Subject ("Ich") + Verb ("ging") + Prepositional Phrase ("in die Bibliothek"). Therefore, "Ich ging in die Bibliothek" conveys that the speaker went to the library at some point in the past.


Example 2:

Sentence: Warum bist du heute nicht gekommen?
Translation: Why didn't you come today?

"Warum"

Translation: Why
Grammatical Aspect: Interrogative adverb
Function: Used to ask for the reason or cause of an action or situation
Description: "Warum" is used at the beginning of a question to inquire about the reason or purpose behind the action being discussed. It sets the sentence up as a question.
"bist"

Translation: are
Grammatical Aspect: Verb
Tense: Present
Mood: Indicative
Voice: Active
Person: Second person singular (du)
Verb Type: Auxiliary verb (part of the present perfect tense construction)
Description: "bist" is the second person singular form of the verb "sein" (to be). It is used as an auxiliary verb in the present perfect tense to form the compound tense with the past participle "gekommen."
"du"

Translation: you
Grammatical Aspect: Pronoun
Case: Nominative (subject of the sentence)
Number: Singular
Person: Second person
Description: "du" is the nominative case pronoun used to indicate the person being addressed in the sentence. It is the subject of the auxiliary verb "bist."
"heute"

Translation: today
Grammatical Aspect: Adverb
Function: Indicates the time when the action is supposed to take place
Description: "heute" is a temporal adverb that specifies the time frame of the action. It indicates that the action in question was expected to occur today.
"nicht"

Translation: not
Grammatical Aspect: Adverb
Function: Negates the verb in the sentence
Description: "nicht" is used to negate the verb phrase. In this sentence, it negates the action of "gekommen" (come), indicating that the action did not take place.
"gekommen"

Translation: come
Grammatical Aspect: Verb
Tense: Present perfect (used with auxiliary "sein")
Mood: Indicative
Voice: Active
Form: Past participle of "kommen" (to come)
Description: "gekommen" is the past participle of the verb "kommen." It is used with the auxiliary verb "bist" to form the present perfect tense, indicating that the action of coming has not occurred up to the present time.
Full Explanation
Sentence Structure: The sentence "Warum bist du heute nicht gekommen?" follows the typical German word order for questions with a question word: interrogative adverb ("Warum") + verb ("bist") + subject ("du") + time adverb ("heute") + negation ("nicht") + past participle ("gekommen").
Interrogative Construction: "Warum" initiates the question by asking for the reason behind the action.
Tense and Aspect: The present perfect tense is used to indicate that the action of coming did not happen by the time of speaking. The auxiliary verb "bist" and the past participle "gekommen" together form this tense.
Negation: The adverb "nicht" negates the action, indicating that the expected action did not occur.
When the sentence components come together, the structure in German follows this pattern: Interrogative Adverb ("Warum") + Auxiliary Verb ("bist") + Subject ("du") + Time Adverb ("heute") + Negation ("nicht") + Past Participle ("gekommen"). Therefore, "Warum bist du heute nicht gekommen?" asks why the addressed person did not come today.

"""

german_examples_words = None

example_dict = {}
example_dict['minimal'] = {}

example_dict['minimal']['sentence'] = {'hindi': hindi_examples_sentences, 'farsi': farsi_examples_sentences, 'german': german_examples_sentences}
example_dict['minimal']['word'] = {'hindi': hindi_examples_words, 'farsi': farsi_examples_words, 'german': german_examples_words}

example_dict['detailed'] = {}
example_dict['detailed']['sentence'] = {'hindi': hindi_examples_sentences_detailed, 'farsi': None, 'german': german_examples_sentences_detailed}
example_dict['detailed']['word'] = {'hindi': None, 'farsi': farsi_examples_words_detailed, 'german': None}
