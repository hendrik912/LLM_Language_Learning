Project using a Large Language Model for language learning

- Clean corpus of sentences of target language (hindi, farsi)
- Sort sentences based on frequency of words
   - word frequency, sentence structure frequency, sentence length, etc.
   
- Use LLM to generate in depth explanations of each sentence via few shot prompting.
  Example output: 
  > The given Hindi sentence 'वह शहर में नहीं है' can be broken down as follows:
  >1. 'वह' (vah) is a pronoun meaning 'he.'
  >2. 'शहर' (shahar) is a noun meaning 'city.'
  >3. 'में' (men) is a preposition meaning 'in.'
  >4. 'नहीं' (nahin) is a negation particle, which negates the following verb.
  >5. 'है' (hai) is the present tense form of the verb 'होना' (hona - to be), which is used to indicate the presence or existence of the subject. The negation of this verb indicates the absence or non-existence of the subject in the given location.

- Export sentences (target language, translation, explanation) to Anki-deck including sound files
