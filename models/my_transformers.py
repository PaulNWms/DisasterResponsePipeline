import nltk
import re
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    tagged_tokens = pos_tag(tokens)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    for tagged_token in tagged_tokens:
        token = tagged_token[0]
        tag = tagged_token[1][0].lower()
        if tag in ('a', 'n', 'r', 'v'):
            clean_tok = lemmatizer.lemmatize(token, pos=tag)
        else:
            clean_tok = lemmatizer.lemmatize(token)

        clean_tokens.append(clean_tok)
        
    stemmed_tokens = [PorterStemmer().stem(w) for w in clean_tokens]
    return stemmed_tokens

def tokenize2(text):
    return tokenize

class TokenCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def my_operation(self, text):
        return len(word_tokenize(text))

    def transform(self, X):
        result = [self.my_operation(text) for text in X]
        return pd.DataFrame(result, columns=['tok_cnt'])
    
class UpperCaseExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def my_operation(self, text):
        length = len(text.strip())
        if length == 0:
            return 0
        uppercase = sum(1 for c in text if c.isupper())
        return uppercase / length
    
    def transform(self, X):
        result = [self.my_operation(text) for text in X]
        return pd.DataFrame(result, columns=['upper_pct'])

class EntityCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def extract_entity_counts_helper(self, t):
        if hasattr(t, 'label') and t.label:
            if t.label() == 'GPE':
                self.gpe += 1
            elif t.label() == 'PERSON':
                self.person += 1
            elif t.label() == 'ORGANIZATION':
                self.org += 1

            for child in t:
                self.extract_entity_counts_helper(child)

    def extract_entity_counts(self, message):
        self.gpe = self.person = self.org = 0
        tree = ne_chunk(pos_tag(word_tokenize(message)))
        self.extract_entity_counts_helper(tree)
        
    def my_operation(self, text):
        self.extract_entity_counts(text)
        return self.gpe, self.person, self.org

    def transform(self, X):
        result = [self.my_operation(text) for text in X]
        return pd.DataFrame(result, columns=['gpe_cnt', 'person_cnt', 'org_cnt'])

