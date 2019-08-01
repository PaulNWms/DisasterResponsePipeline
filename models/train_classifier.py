import sys
import re
import pandas as pd
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.sem import relextract
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

import sys
sys.path.append('../models')
from my_transformers import tokenize, TokenCountExtractor, UpperCaseExtractor, EntityCountExtractor

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

def load_data(database_filepath):
    """Loads data from a database file into X and Y matrices.
        :param database_filepath: the database file to load
        :type database_filepath: str
        :returns: X and Y tables, and a list of category names
        :rtype: tuple of (DataFrame, DataFrame, list)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    # X is a dataframe with 1 column - the messages
    X = df.message
    # Y is all the categories, after a little cleaning
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # categories are derived from the column names
    category_names = list(Y)[1:]
    return X, Y, category_names

def build_model():
    """Assemble the model.
        :returns: a pipeline of estimators
        :rtype: Pipeline
    """
    return Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            # custom tranformers, see my_transformers.py
            ('tok_cnt', TokenCountExtractor()),
            ('upper_pct', UpperCaseExtractor()),
            ('ent_cnt', EntityCountExtractor()),
        ])),

        ('clf', MultiOutputClassifier(MultinomialNB())),
    ])



def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model accuracy using the supplied test set
       and prints the results.
        :param model: the trained model to evaluate
        :type model: Pipeline
        :param X_test: test messages
        :type X_test: DataFrame
        :param Y_test: test categories
        :type Y_test: DataFrame
    """
    Y_pred = model.predict(X_test)
    indices = range(Y_test.shape[1])
    for col, col_name in zip(indices, category_names):
        test = Y_test.values[:, col]
        pred = Y_pred[:, col]
        sep = col_name + ' -----------------------------------------'
        rpt = classification_report(test, pred)
        print(sep)
        print(rpt)


def save_model(model, model_filepath):
    """Saves the model to a pickle file.
        :param model: the trained model to save
        :type model: Pipeline
        :param model_filepath: the path to the pickle file
        :type model_filepath: str
    """
    pipeline_pkl = open(model_filepath, 'wb')
    pickle.dump(model, pipeline_pkl)
    pipeline_pkl.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()