import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
from corextopic import vis_topic as vt


def all_pickles_to_1(pickle_dir):
    ''' Input directory of processed podcast tweet dataframes
        Returns all tweets in single dataframe
    '''
    pickle_list = os.listdir(pickle_dir)
    pickle_list = sorted(pickle_list)

    temp_df = pd.read_pickle(pickle_dir + pickle_list[0])
    all_tweets = temp_df

    for pdcst in pickle_list[1:]:
        temp_df = pd.read_pickle(pickle_dir + pdcst)
        all_tweets = all_tweets.append(temp_df, ignore_index=True)

    return all_tweets


pickle_dir = './tweet_pickles/'
num_topics = 20
num_max_df = 0.5
num_min_df = 100

print('Loading tweets...', end='')
all_tweets = all_pickles_to_1(pickle_dir)
print('Done')

# Model Parameters
vectorizer = TfidfVectorizer(
    strip_accents='ascii',
    encoding='unicode',
    max_df=num_max_df,
    min_df=num_min_df,
    max_features=None,
    ngram_range=(1, 2),
    norm=None,
    binary=True,
    use_idf=False,
    sublinear_tf=False)

model = ct.Corex(
    n_hidden=num_topics,
    verbose=1,
    max_iter=2,
    seed=42)

# pipeline = Pipeline(
#     [(), ()]
# )

print('Vectorizing tweets...', end='')
vect_fit = vectorizer.fit(all_tweets['text'])
tfidf = vectorizer.transform(all_tweets['text'])
vocab = vect_fit.get_feature_names()
print('Done')

print('Fitting CorEx model...')
anchors = []
model = model.fit(
    tfidf,
    words=vocab)
print('Done')

vt.vis_rep(model, column_label=vocab,
           prefix='./corex_models/{}-topic-model'.format(num_topics))