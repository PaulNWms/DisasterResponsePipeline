import nltk
import re
import pandas as pd
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

def tokenize(text):
    """Tokenize a string.
        :param text: the text to be tokenized
        :type text: str
        :returns: a list of token strings
        :rtype: list
    """
    # Get rid of URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize and tag
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatize
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
    
    # Stem
    stemmed_tokens = [PorterStemmer().stem(w) for w in clean_tokens]
    return stemmed_tokens

class TokenCountExtractor(BaseEstimator, TransformerMixin):
    """Transformer that count the tokens in each cell of a DataFrame colu        """
    def fit(self, X, y=None):
        """Unused.
        """
        return self
    
    def my_operation(self, text):
        """Counts the tokens in a string.
            :param text: the text to be count
            :type text: str
            :returns: the number of tokens
            :rtype: int
        """
        return len(word_tokenize(text))

    def transform(self, X):
        """Transforms a text column into a column of token counts.
            :param X: the text to be count
            :type X: Series
            :returns: the number of tokens
            :rtype: DataFrame of dimension [-1, 1]
        """
        result = [self.my_operation(text) for text in X]
        return pd.DataFrame(result, columns=['tok_cnt'])
    
class UpperCaseExtractor(BaseEstimator, TransformerMixin):
    """Transformer that counts the number of upper case letters 
       in each cell of a DataFrame column / total length.
    """
    def fit(self, X, y=None):
        """Unused.
        """
        return self

    def my_operation(self, text):
        """Counts the number of upper case letters in a string.
            :param text: the text to be counted
            :type text: str
            :returns: the number of upper case letters
            :rtype: int
        """
        length = len(text.strip())
        if length == 0:
            return 0
        uppercase = sum(1 for c in text if c.isupper())
        return uppercase / length
    
    def transform(self, X):
        """Transforms a text column into a column of upper case letter percent.
            :param X: the text to be counted
            :type X: Series
            :returns: the number of upper case letters
            :rtype: DataFrame of dimension [-1, 1]
        """
        result = [self.my_operation(text) for text in X]
        return pd.DataFrame(result, columns=['upper_pct'])

class EntityCountExtractor(BaseEstimator, TransformerMixin):
    """Transformer that counts the named entities
       in each cell of a DataFrame column.
    """
    def fit(self, X, y=None):
        """Unused.
        """
        return self
    
    def extract_entity_counts_helper(self, t):
        """Traverses a sentence diagram tree and counts the named entites.
           The intermediate results are accumulated in instance variables.
            :param t: tree to be counted
            :type t: tree
        """
        if hasattr(t, 'label') and t.label:
            if t.label() == 'GPE':
                self.gpe += 1
            elif t.label() == 'PERSON':
                self.person += 1
            elif t.label() == 'ORGANIZATION':
                self.org += 1

            # recurse
            for child in t:
                self.extract_entity_counts_helper(child)

    def extract_entity_counts(self, message):
        """Count the named entites in a message.
           The intermediate results are accumulated in instance variables.
            :param message: text to be analyzed
            :type message: str
        """
        self.gpe = self.person = self.org = 0
        tree = ne_chunk(pos_tag(word_tokenize(message)))
        self.extract_entity_counts_helper(tree)
        
    def my_operation(self, text):
        """Count the named entites in a message.
            :param message: text to be analyzed
            :type message: str
            :returns: the number of named entities in a tuple
            :rtype: tuple of (gpe, person, org) counts
        """
        self.extract_entity_counts(text)
        return self.gpe, self.person, self.org

    def transform(self, X):
        """Transforms a text column into 3 columns of named entity counts.
            :param X: the text to be counted
            :type X: Series
            :returns: the number of named entities
            :rtype: DataFrame of dimension [-1, 3]
        """
        result = [self.my_operation(text) for text in X]
        return pd.DataFrame(result, columns=['gpe_cnt', 'person_cnt', 'org_cnt'])

