import os
import re
import emoji
import regex
import numpy as np
import preprocessor as p

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer


data_dir = '/home/ed/github/pod_tweets/follower_twts/'
categories = os.listdir(data_dir)
# print(categories)
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.EMOJI)
stemmer = SnowballStemmer('english')
punct_str = '''!"$%&'()*+,-./:;<=>?[\]^_`{|}~’'''
# stop_words = STOPWORDS
stop_words = STOPWORDS.union(set(['', 'ive', 'im', 'amp']))
# Emoji patterns
emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map sym
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              "]+", flags=re.UNICODE)


def lemmatize_stemming(text):
    '''
    '''
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def split_count(text):
    '''
    '''
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list


def find_hash(text):
    '''
    '''
    hashtag_list = []
    data = regex.findall(r'#\w*', text)
    for word in data:
        hashtag_list.append(word)
    return hashtag_list


def find_mention(text):
    '''
    '''
    mention_list = []
    data = regex.findall(r'@\w*', text)
    for word in data:
        mention_list.append(word)
    return mention_list


def my_preprocess(text):
    '''
    '''
    doc_emoji = split_count(text)
    doc_hash = find_hash(text)
    doc_mentions = find_mention(text)
    text = text.lower()
    text = text.replace('\\n',' ')
    text = p.clean(text)
    text = text.translate(str.maketrans(' ', ' ', punct_str))
    text = re.sub(r' \d+ ', ' ', text)
    text = re.sub(r' \d+ ', ' ', text)
    words = []
    for word in text.split(' '):
        words.append(word)
    words = [w for w in words if not w in stop_words]
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words and len(token) > 3:
            result.append(lemmatize_stemming(token))

    result = result + doc_emoji + doc_hash + doc_mentions
    tweet_txt = ' '.join(result)
    return tweet_txt


docs_to_train = sklearn.datasets.load_files(data_dir, description=None,
                                            categories=categories,
                                            load_content=True,
                                            encoding='utf-8', shuffle=True,
                                            random_state=42)
print('Docs Loaded...')

X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data,
                                                    docs_to_train.target,
                                                    test_size=0.1)

print('Processing...', end='')
reformat = lambda x: my_preprocess(x)
X_train = list(map(reformat, X_train))
print('...', end='')
X_test = list(map(reformat, X_test))
print('...Done')

vectorizer = CountVectorizer(token_pattern=r'[^\s]+', encoding='unicode',
                             lowercase=None, strip_accents=None,
                             stop_words=None)

text_clf = Pipeline(
    [('vect', vectorizer),
     ('tfidf', TfidfTransformer(use_idf=True)),
     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                           random_state=42, verbose=1)), ])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print(np.mean(predicted == y_test))

print(metrics.classification_report(y_test, predicted,
                                    target_names=docs_to_train.target_names))
