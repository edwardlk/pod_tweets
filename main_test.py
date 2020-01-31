import time
import tweepy
import numpy as np
import pandas as pd
from twitter_funcs import(get_x_tweets, get_all_followers, search_for_pod)


# Search for screen name of all podcasts
# pod_list_df = pd.read_csv('pod_list.csv', index_col=0)
#
# search_results_df = pd.DataFrame(columns=['search', 'result_num', 'screen_name', 'user_id', 'followers_count', 'Description'])
#
# for x in range(1000, 2000):
#     search_results_df = search_results_df.append(
#          pd.DataFrame(search_for_pod(pod_list_df.at[x, 'Name'], 1),
#                       columns=['search', 'result_num', 'screen_name', 'user_id', 'followers_count', 'Description']),
#                       ignore_index=True
#     )
#     if x % 10 == 0:
#         print('{} of {} complete'.format(x, len(pod_list_df)))
#     if x % 10 == 0:
#         search_results_df.to_csv('search_results.csv', index=False)
#     time.sleep(1.2)
#
# search_results_df.to_csv('search_results.csv', index=False)

# Create df of podcast followers
resource_dir = '/home/ed/github/pod_tweets/'
data_file = 'podchaserJan30fixed.csv'
pod_screenName = pd.read_csv(resource_dir + data_file, index_col=0)

idx_list = pod_screenName.index.tolist()

for x in idx_list[84:]:
    if not pd.isnull(pod_screenName.loc[x, 'twitter_s_n']):
        temp_df = pd.DataFrame(get_all_followers(
                     pod_screenName.at[x, 'twitter_s_n']), columns=['user_id'])
        fname = ''.join(e for e in pod_screenName.at[x, 'Name'] if e.isalnum())
        temp_df.to_csv('follower_ids/{}.csv'.format(fname), index=False)
        print('{} files done'.format(x))

# Create df of podcast followers' latest tweets
# follow_ids = 'follower_ids/BehindtheBastards.csv'
# follow_twts = 'follower_twts/BehindtheBastards.csv'
# follow_ids = 'follower_ids/SeincastASeinfeldPodcast.csv'
# follow_twts = 'follower_twts/SeincastASeinfeldPodcast.csv'
follow_ids = 'follower_ids/BehindtheBastards.csv'
follow_twts = 'follower_twts/BehindtheBastards.csv'

# temp_df = pd.read_csv(follow_ids)
# temp_df['tweets'] = [[]]*len(temp_df)
#
# for x in range(len(temp_df)):
#     try:
#         user_tweet_list = get_x_tweets(temp_df.at[x, 'user_id'], 10)
#     except tweepy.TweepError:
#         user_tweet_list = []
#         print("Failed to run the command on that user, Skipping...")
#     temp_df.at[x, 'tweets'] = user_tweet_list
#
#     if x % 10 == 0:
#         print('{} users done'.format(x))
#     if x % 100 == 0:
#         temp_df.to_csv(follow_twts, index=False)
#     time.sleep(0.700)
#
# temp_df.to_csv(follow_twts, index=False)
