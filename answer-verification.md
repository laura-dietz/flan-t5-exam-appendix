# Answer Verification

We are verifying the correctness of a predicted answer against a set of  known correct answers (usually a set of one element)

Both answers are normalized with `stop_stem_normalize_answer`, then:

* if `len(stemmed_answer) <2`,  unnormalized answers are verified with exact match
* if normalized answers match, this counts as a match
* else, and if `len(stemmed_answer)>=4`,  a string edit distance match  `fuzz.ratio(stemmed_answer, stemmed_gold) > 80 ` is verifed.


Additionally there is some special handling of true/false questions, as given below.



##  Normalization and Stringedit distance check

```python

import nltk
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')  

    def stop_stem_normalize_answer(self, text:str)->str:
        # Convert text to lowercase
        text = text.lower().strip()


        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]

        # Rejoin words
        normalized_text = ' '.join(stemmed_tokens)

        return normalized_text


    def check_answer_stemmed(self,answer:str)->Optional[bool]:
        stemmed_answer = self.stop_stem_normalize_answer(answer)
        if len(stemmed_answer) >=2:
            is_match = stemmed_answer in self.stop_stemmed_correct_answers
            if is_match: 
                return is_match

        if len(stemmed_answer) >=4:
            is_fuzzy = any (fuzz.ratio(stemmed_answer, stemmed_gold) > 80 for stemmed_gold in self.stop_stemmed_correct_answers)
            return is_fuzzy
        return None

```

## True/false questions


```python

    def check_true_false(self, answer:str)->Optional[bool]:
        FALSE_answers = {"no", "incorrect","false"}
        TRUE_answers = {"yes", "correct","true"}
        answer_ = self.correct.lower().strip()

        if answer_ == "false":
            if answer.lower() in FALSE_answers:
                return True
            else:
                return False

        if answer_ == "true":
            if answer.lower() in TRUE_answers:
                return True
            else:
                return False
            
        return None
            
```
