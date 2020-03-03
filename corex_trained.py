import os
import pandas as pd
import joblib

# from sklearn.metrics import make_scorer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
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
        all_tweets = all_tweets.append(temp_df, ignore_index=True, sort=True)

    return all_tweets


resource_dir = './resources/'
pickle_dir = './tweet_pickles/'
num_max_df = 0.7
num_min_df = 10
vocab = joblib.load('./resources/vocab_dump.joblib')

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
    sublinear_tf=False,
    stop_words='english',
    vocabulary=vocab)

do_vect = True
print('Loading tweets...', end='')
all_tweets = all_pickles_to_1(pickle_dir)
print('Done')

if do_vect:
    print('Vectorizing tweets...', end='')
    vect_fit = vectorizer.fit(all_tweets['text'])
    tfidf = vectorizer.transform(all_tweets['text'])
    vocab = vect_fit.get_feature_names()
    print('Done')
    joblib.dump(vect_fit, './resources/all_vect_fit_dump.joblib')
    joblib.dump(tfidf, './resources/all_tfidf_dump.joblib')
    joblib.dump(vocab, './resources/all_vocab_dump.joblib')
else:
    print('Loading vects...', end='')
    vect_fit = joblib.load('./resources/all_vect_fit_dump.joblib')
    tfidf = joblib.load('./resources/all_tfidf_dump.joblib')
    print('Done')

model = joblib.load('./resources/model_dump.joblib')

topic_df = pd.DataFrame(
    model.predict(tfidf),
    columns=["topic_{}".format(i+1) for i in range(100)]).astype(float)
topic_df.index = all_tweets.index
all_df = pd.concat([all_tweets, topic_df], axis=1)

all_df.to_csv(resource_dir + 'all_df.csv')

vt.vis_rep(model, column_label=vocab,
           prefix='./corex_models/trained-topic-model')

# topics = model.get_topics()
# for topic_n, topic in enumerate(topics):
#     words, mis = zip(*topic)
#     topic_str = str(topic_n+1)+': '+','.join(words)
#     print(topic_str)
