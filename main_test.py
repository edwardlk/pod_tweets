import time
import tweepy
import numpy as np
import pandas as pd
from twitter_funcs import(get_x_tweets,
                          get_all_followers)


podcasts_list = ['knowledge_fight']  # , 'SlatePodcasts']

# Create df of podcast followers
for pdcst in podcasts_list:
    temp_df = pd.DataFrame(get_all_followers(pdcst), columns=['user_id'])
    temp_df.to_csv('follow_{}_ids.csv'.format(pdcst), index=False)

# Create df of podcast followers' latest tweets
temp_df['tweets'] = [[]]*len(temp_df)

for x in range(len(temp_df)):
    try:
        user_tweet_list = get_x_tweets(temp_df.at[x, 'user_id'], 10)
    except tweepy.TweepError:
        user_tweet_list = []
        print("Failed to run the command on that user, Skipping...")
    temp_df.at[x, 'tweets'] = user_tweet_list

    if x % 10 == 0:
        print('{} users done'.format(x))
    if x % 100 == 0:
        temp_df.to_csv('follow_{}_tweets.csv'.format('knowledge_fight'), index=False)
    time.sleep(1)

temp_df.to_csv('user_tweets.csv', index=False)
