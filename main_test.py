import os
import time
import datetime
import tweepy
import numpy as np
import pandas as pd
from twitter_funcs import(get_x_tweets, get_all_followers,
                          search_for_pod, get_pod_tweets, get_user_info)


resource_dir = './resources/'

# Search for screen name of all podcasts
pod_list_df = pd.read_csv('pod_list.csv', index_col=0)

search_results_df = pd.DataFrame(columns=['search', 'result_num', 'screen_name', 'user_id', 'followers_count', 'Description'])

for x in range(1000, 2000):
    search_results_df = search_results_df.append(
         pd.DataFrame(search_for_pod(pod_list_df.at[x, 'Name'], 1),
                      columns=['search', 'result_num', 'screen_name', 'user_id', 'followers_count', 'Description']),
                      ignore_index=True
    )
    if x % 10 == 0:
        print('{} of {} complete'.format(x, len(pod_list_df)))
    if x % 10 == 0:
        search_results_df.to_csv('search_results.csv', index=False)
    time.sleep(1.2)

search_results_df.to_csv('search_results.csv', index=False)

# Create df of podcast followers
data_file = 'podchaserJan30fixed.csv'
pod_screenName = pd.read_csv(resource_dir + data_file, index_col=0)

idx_list = pod_screenName.index.tolist()

for x in idx_list[180:]:
    if not pd.isnull(pod_screenName.loc[x, 'twitter_s_n']):
        temp_df = pd.DataFrame(get_all_followers(
                     pod_screenName.at[x, 'twitter_s_n']), columns=['user_id'])
        fname = ''.join(e for e in pod_screenName.at[x, 'Name'] if e.isalnum())
        temp_df.to_csv('follower_ids/{}.csv'.format(fname), index=False)
        print('{} files done'.format(x))

# Create df of podcast followers' latest tweets
pods = os.listdir('./follower_ids/')
pods = sorted(pods)
start_time = time.time()

for pdcst in pods:
    follow_ids = 'follower_ids/' + pdcst
    follow_twts = 'follower_twts/' + pdcst[:-4] + '-200.csv'

    temp_df = pd.read_csv(follow_ids)
    temp_df['tweets'] = [[]]*len(temp_df)

    users_to_scrape = 1000
    if len(temp_df) < 1000:
        users_to_scrape = len(temp_df)

    print('Gathering {} followers...'.format(pdcst[:-4]))
    for x in range(users_to_scrape):
        try:
            user_tweet_list = get_x_tweets(temp_df.at[x, 'user_id'], 200)
        except tweepy.TweepError:
            user_tweet_list = []
            print("Failed to run the command on that user, Skipping...")
        temp_df.at[x, 'tweets'] = user_tweet_list

        if x % 50 == 0:
            print('{} users done'.format(x))
        if x % 100 == 0:
            temp_df.to_csv(follow_twts, index=False)
        time.sleep(0.3)

    temp_df.to_csv(follow_twts, index=False)

    total_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('It took {}.'.format(total_time))

# Get podcasts' latest tweets
pod_list = 'pod-twitter_list2.csv'

pod_list_df = pd.read_csv(resource_dir + pod_list)

pod_df = pd.DataFrame(columns=['podcast', 'time', 'tweets'])

startT = time.time()
for pdcst in pod_list_df['twitter_s_n']:
    print('working on {}...'.format(pdcst), end='')
    temp = {}
    temp['podcast'] = pdcst
    temp['time'], temp['tweets'] = get_pod_tweets(pdcst)
    df = pd.DataFrame(temp)
    pod_df = pod_df.append(df, ignore_index=True)
    print('Done')
    time.sleep(1)

total_time = str(datetime.timedelta(seconds=int(time.time()-startT)))
print('It took {}.'.format(total_time))

pod_df.to_csv(resource_dir + 'test_df.csv')

# Get User user_info
pod_list = 'pod-twitter_list2.csv'

pod_list_df = pd.read_csv(resource_dir + pod_list)

df_cols = ['podcast_name', 'id_str', 'name', 'screen_name', 'description',
           'followers_count', 'profile_image_url_https']

user_info_df = pd.DataFrame(columns=df_cols)

for x in range(len(pod_list_df)):
    temp = [pod_list_df.loc[x, 'Name']]

    temp2 = get_user_info(pod_list_df.loc[x, 'twitter_s_n'])

    temp.extend(temp2)

    user_info_df.loc[x] = temp

print(user_info_df)
user_info_df.to_csv('pod_twitter_info.csv')

user_info_df = pd.read_csv(resource_dir + 'pod_twitter_info.csv', index_col=0)
follow_order = user_info_df.sort_values(by=['followers_count'], ascending=False).index.tolist()

follow_count_dict = {}
for x in follow_order:
    key_txt = user_info_df.loc[x, 'podcast_name']
    key_txt = ''.join(e for e in key_txt if e.isalnum())
    follow_count_dict[key_txt] = user_info_df.loc[x, 'followers_count']

print(follow_count_dict)
# print(user_info_df[['podcast_name', 'followers_count']].head(5))
