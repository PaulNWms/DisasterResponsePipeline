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

import my_transformers

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///pauls_messages.db')
    df = pd.read_sql_table('messages', engine)
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y)[1:]
    return X, Y, category_names

def build_model():
    return Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('tok_cnt', TokenCountExtractor()),
            ('upper_pct', UpperCaseExtractor()),
            ('ent_cnt', EntityCountExtractor()),
        ])),

        ('clf', MultiOutputClassifier(MultinomialNB())),
    ])



def evaluate_model(model, X_test, Y_test, category_names):
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