import re
import os
import time
import regex
import emoji
import pandas as pd
import numpy as np
import preprocessor as p
from resources.stopwords import my_stopwords


resource_dir = '/home/ed/github/pod_tweets/resources/'
tweets_dir = '/home/ed/github/pod_tweets/follower_twts/'

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.EMOJI)
punct_str = '''!"$%&'()*+,-./:;<=>?[\]^_`{|}~’'''
stop_words = my_stopwords


def find_emoji(text):
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


def tweet_split_2_process(user_id, tweets_str):
    '''
    '''
    df_out = pd.DataFrame(columns=['user_id'], dtype=np.int32)
    output_txt = ''
    output_emoji = ''
    output_hash = ''
    output_mention = ''

    tweets_str = tweets_str.replace('''', "''', "', '")
    tweets_str = tweets_str.replace('''", ''''', "', '")
    tweets_str = tweets_str.replace('''", "''', "', '")
    tweet_list = tweets_str.split("', '")

    for tweet in tweet_list:
        tweet_p = my_preprocess(tweet)
        output_txt = output_txt + ' ' + tweet_p[0]
        if tweet_p[1] != []:
            output_emoji = output_emoji + ' ' + ' '.join(tweet_p[1])
        if tweet_p[2] != []:
            output_hash = output_hash + ' ' + ' '.join(tweet_p[2])
        if tweet_p[3] != []:
            output_mention = output_mention + ' ' + ' '.join(tweet_p[3])

    df_out.loc[0, 'user_id'] = user_id
    df_out.astype({'user_id': np.int32})
    df_out.loc[0, 'text'] = output_txt
    df_out.loc[0, 'emojis'] = output_emoji
    df_out.loc[0, 'hashtags'] = output_hash
    df_out.loc[0, 'mentions'] = output_mention

    return df_out


def tweet_split_2_process_v2(user_id, time, tweets_str):
    '''['podcast', 'time', 'text', 'emojis', 'hashtags', 'mentions']
    '''
    df_out = pd.DataFrame(columns=['podcast'])
    output_txt = ''
    output_emoji = ''
    output_hash = ''
    output_mention = ''

    tweets_str = tweets_str.replace('''', "''', "', '")
    tweets_str = tweets_str.replace('''", ''''', "', '")
    tweets_str = tweets_str.replace('''", "''', "', '")
    tweet_list = tweets_str.split("', '")

    for tweet in tweet_list:
        tweet_p = my_preprocess(tweet)
        output_txt = output_txt + ' ' + tweet_p[0]
        if tweet_p[1] != []:
            output_emoji = output_emoji + ' ' + ' '.join(tweet_p[1])
        if tweet_p[2] != []:
            output_hash = output_hash + ' ' + ' '.join(tweet_p[2])
        if tweet_p[3] != []:
            output_mention = output_mention + ' ' + ' '.join(tweet_p[3])

    df_out.loc[0, 'podcast'] = user_id
    df_out.loc[0, 'time'] = time
    df_out.loc[0, 'text'] = output_txt
    df_out.loc[0, 'emojis'] = output_emoji
    df_out.loc[0, 'hashtags'] = output_hash
    df_out.loc[0, 'mentions'] = output_mention

    return df_out


def my_preprocess(text):
    '''
    '''
    doc_emoji = find_emoji(text)
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
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in stop_words:  # and len(token) > 3:
#             result.append(token)     # lemmatize_stemming(token))
#     result = result + doc_emoji + doc_hash + doc_mentions
    result = words + doc_emoji + doc_hash + doc_mentions
    tweet_txt = ' '.join(result)
#     return tweet_txt
    return ' '.join(words), doc_emoji, doc_hash, doc_mentions


X = os.listdir('./pre-processed_tweets/')
X = sorted(X)

for pdcst in X:
    print('Converting {}...'.format(pdcst[:-4]), end='')
    pdcst_tweet_df = pd.read_csv('./pre-processed_tweets/' + pdcst)
    tweet_split_df = pd.DataFrame(columns=['user_id', 'text', 'emojis', 'hashtags', 'mentions'])
    for x in range(len(pdcst_tweet_df)):
        if pdcst_tweet_df.loc[x, 'tweets'] != '[]':
            temp_df = tweet_split_2_process(pdcst_tweet_df.loc[x, 'user_id'], pdcst_tweet_df.loc[x, 'tweets'])
            tweet_split_df = tweet_split_df.append(temp_df, ignore_index=True)
    tweet_split_df['podcast_name'] = pdcst[:-4]
    print('Saving...', end='')
    tweet_split_df.to_pickle('./processing_output/' + pdcst[:-4] + '.pkl')
    print('Done')
