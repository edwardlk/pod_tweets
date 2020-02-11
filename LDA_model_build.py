import os
import re
import sklearn
import regex
import emoji
import pandas as pd
import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.model_selection import GridSearchCV
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from joblib import dump, load
from datetime import datetime


n_samples = 2000
n_features = 1000
n_components = 30
n_top_words = 5

stop_words = STOPWORDS.union(
    set(['', 'ive', 'im', 'amp', 'like', 'fuck', 'shit']))

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.EMOJI)
stemmer = SnowballStemmer('english')
punct_str = '''!"$%&'()*+,-./:;<=>?[\]^_`{|}~’'''


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
    text = text.replace('\\n', ' ')
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


def tweet_split_2_process(tweets_str):
    '''
    '''
    output = []
    tweet_list = tweets_str.split(', ')
    for tweet in tweet_list:
        tweet_p = my_preprocess(tweet)
        output.append(tweet_p)

    return output


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


data_dir = './follower_twts/'
resources_dir = './resources/'
categories = os.listdir(data_dir)

have_file_dump = True
have_model_dump = True
file_dump_name = '202002102056_tweet_p_list.joblib'
model_dump = '202002110724_LDA_Grid.joblib'

if not have_file_dump:
    print('Loading dataset...', end='')
    docs_to_train = sklearn.datasets.load_files(
        data_dir, description=None, categories=categories, load_content=True,
        encoding='utf-8', shuffle=True, random_state=42)
    print('Done')
    tweet_txt = docs_to_train['data']

    print('Splitting tweets...', end='')
    reformat = lambda x: tweet_split_2_process(x)
    tweet_txt_p = list(map(reformat, tweet_txt))
    tweet_p_list = []
    for sublist in tweet_txt_p:
        tweet_p_list.extend(sublist)
    print('Done')

    print('Dumping formatted tweets...', end='')
    dump1 = (resources_dir + '{}_tweet_p_list.joblib'.format(datetime.now().strftime("%Y%m%d%H%M")))
    dump(tweet_p_list, dump1)
    print('Done')
else:
    load_file = resources_dir + file_dump_name
    tweet_p_list = load(load_file)

print('Extracting tf features for LDA...', end='')
tf_vectorizer = CountVectorizer(
    token_pattern=r'[^\s]+', encoding='unicode', lowercase=None, strip_accents=None, stop_words=None,
    max_df=0.95, min_df=2, max_features=n_features)

tf = tf_vectorizer.fit_transform(tweet_p_list)
print('Done')

if not have_model_dump:
    # Define Search Param
    search_params = {'n_components': [40], 'learning_decay': [0.9]}

    # Init the Model
    lda = LatentDirichletAllocation(
        n_components=n_components, max_iter=5, learning_method='online',
        learning_offset=50., random_state=0, n_jobs=-1)

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params, n_jobs=-1, verbose=2)

    # Do the Grid Search
    print('Beginning the grid search...', end='')
    model.fit(tf)
    print('Done')

    dump2 = (resources_dir + '{}_LDA_Grid.joblib'.format(datetime.now().strftime("%Y%m%d%H%M")))
    dump(model, dump2)
else:
    model_file = resources_dir + model_dump
    model = load(model_file)

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(tf))

topic_names = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
topic_matrix = pd.DataFrame(columns=topic_names)

categories_list = docs_to_train.target_names

for x1 in range(len(categories_list)):
    subdocs_to_train = sklearn.datasets.load_files(
        data_dir, description=None, categories=categories_list[x1],
        load_content=True, encoding='utf-8', shuffle=True, random_state=42)

    subset_tweet_txt = subdocs_to_train['data']

    print("Extracting tf features for LDA of {}...".format(categories_list[x1]))
    sub_tf_vectorizer = CountVectorizer(
        token_pattern=r'[^\s]+', encoding='unicode', lowercase=None, strip_accents=None, stop_words=None,
        max_features=n_features)

    reformat = lambda x: tweet_split_2_process(x)
    subset_tweet_txt_p = list(map(reformat, subset_tweet_txt))
    subset_tweet_p_list = []

    for sublist in subset_tweet_txt_p:
        subset_tweet_p_list.extend(sublist)

    corp_to_tf = [' '.join(subset_tweet_p_list)]

    sub_tf = sub_tf_vectorizer.fit_transform(corp_to_tf)

    X = best_lda_model.transform(sub_tf)

    topic_matrix.loc[categories_list[x1]] = X[0]

topic_matrix.to_pickle(resources_dir + 'topic_matrix_{}.pkl'.format(datetime.now().strftime("%Y%m%d%H%M")))


rank_matrix = pd.DataFrame()
for x2 in range(len(topic_matrix)):
    df = pd.DataFrame(topic_matrix.iloc[x2])
    df = df.sort_values(by=[str(df.columns[0])], ascending=False)
    rank_matrix[df.columns[0]] = df.index.tolist()

rank_matrix.to_pickle(resources_dir + 'rank_matrix_{}.pkl'.format(datetime.now().strftime("%Y%m%d%H%M")))
