import os
import pandas as pd
import joblib

# from sklearn.metrics import make_scorer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
from corextopic import vis_topic as vt


pickle_dir = './tweet_pickles/'
num_topics = 100
num_max_df = 0.7
num_min_df = 10


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


def params_search(fit_text, num_max_df, num_min_df, num_topics):
    '''
    '''
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
        seed=42)

    # vectorizer
    vect_fit = vectorizer.fit(fit_text)
    tfidf = vectorizer.transform(fit_text)
    vocab = vect_fit.get_feature_names()

    anchors = []
    model = model.fit(
        tfidf,
        words=vocab)

    vt.vis_rep(model, column_label=vocab,
               prefix='./corex_models/{}-topic-model'.format(num_topics))

    model_tc = model.tc

    vect_print = 'Vect params: min_df={}, max_df={}'.format(num_min_df, num_max_df)
    corex_print = 'CorEx params: n_t={}, tc={}'.format(num_topics, model_tc)

    return vect_print + ' ' + corex_print


print('Loading tweets...', end='')
all_tweets = all_pickles_to_1(pickle_dir)
print('Done')

# for x1 in [0.4, 0.5, 0.6]:
#     print(params_search(all_tweets['text'], x1, num_min_df, num_topics))

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
    sublinear_tf=False,
    stop_words='english')

model = ct.Corex(
    n_hidden=num_topics,
    seed=42)
#
# vect_fit = vectorizer.fit(all_tweets['text'])
# tfidf = vectorizer.transform(all_tweets['text'])
# vocab = vect_fit.get_feature_names()
#
# model = model.fit(tfidf, words=vocab)
#
# # Pipeline for gridsearch
# pipe = Pipeline(
#     [('tfidf', vectorizer),
#      ('corex', model)]
# )
#
# # pipeline.fit(all_tweets['text'])
#
# param_grid = {
#     'tfidf__max_df': [0.3, 0.4, 0.5, 0.6, 0.7],
#     'tfidf__min_df': [10, 30, 100, 300, 1000],
# }
#
# my_scorer = make_scorer(model.tc)
# grid_search = GridSearchCV(pipe, param_grid, scoring=my_scorer, n_jobs=-1)
#
# grid_search.fit(all_tweets['text'])
#
# print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
# print(grid_search.best_params_)

anchors = ['trump',
           ['win', 'giveaway'],
           'vote',
           ['sanders', 'warren', 'biden', 'democratic', 'buttigieg'],
           ['kobe', 'bryant'],
           'book',
           ['super', 'bowl'],
           ['climate', 'change'],
           ['podcast', 'episode'],
           ['mental', 'health'],
           ['health', 'care'],
           'coronavirus',
           'australia',
           'jesus',
           ['sexually', 'assaulted'],
           ['social', 'justice'],
           'brexit',
           ['black', 'history'],
           ['conspiracy', 'theories'],
           ['trans', 'people'],
           'song',
           ['movie', 'film'],
           'food',
           'drink'
           ]
do_vect = True

if do_vect:
    print('Vectorizing tweets...', end='')
    vect_fit = vectorizer.fit(all_tweets['text'])
    tfidf = vectorizer.transform(all_tweets['text'])
    vocab = vect_fit.get_feature_names()
    print('Done')
    joblib.dump(vect_fit, './resources/vect_fit_dump.joblib')
    joblib.dump(tfidf, './resources/tfidf_dump.joblib')
    joblib.dump(vocab, './resources/vocab_dump.joblib')
else:
    vect_fit = joblib.load('./resources/vect_fit_dump.joblib')
    tfidf = joblib.load('./resources/tfidf_dump.joblib')
    vocab = joblib.load('./resources/vocab_dump.joblib')


print('Fitting CorEx model...')
model = model.fit(
    tfidf,
    anchors=anchors,
    anchor_strength=3,
    words=vocab)
print('Done')
joblib.dump(model, './resources/model_dump.joblib')

vt.vis_rep(model, column_label=vocab,
           prefix='./corex_models/{}-topic-model'.format(num_topics))

model_tc = model.tc

vect_print = 'Vect params: min_df={}, max_df={}'.format(num_min_df, num_max_df)
corex_print = 'CorEx params: n_t={}, tc={}'.format(num_topics, model_tc)

print(vect_print + ' ' + corex_print)

topic_df = pd.DataFrame(
    model.transform(tfidf),
    columns=["topic_{}".format(i+1) for i in range(num_topics)]).astype(float)
topic_df.index = all_tweets.index
all_df = pd.concat([all_tweets, topic_df], axis=1)

all_df.to_csv('all_df.csv')

# Train successive layers
print('fit2...')
tm_layer2 = ct.Corex(n_hidden=10)
tm_layer2.fit(model.labels)
joblib.dump(tm_layer2, './resources/tm_layer2_dump.joblib')

vect_print = 'Vect params: min_df={}, max_df={}'.format(num_min_df, num_max_df)
corex_print = 'CorEx params: n_t={}, tc={}'.format(10, tm_layer2.tc)

vt.vis_rep(tm_layer2, column_label=vocab,
           prefix='./corex_models/{}-topic-model'.format(10))

tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels)
joblib.dump(tm_layer3, './resources/tm_layer3_dump.joblib')

vect_print = 'Vect params: min_df={}, max_df={}'.format(num_min_df, num_max_df)
corex_print = 'CorEx params: n_t={}, tc={}'.format(num_topics, model_tc)

vt.vis_rep(tm_layer3, column_label=vocab,
           prefix='./corex_models/{}-topic-model'.format(1))
