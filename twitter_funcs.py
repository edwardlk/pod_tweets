import tweepy
import time

# Twitter API credentials
from auth import (
    api_key,
    api_secret_key,)

# Auths
auth = tweepy.AppAuthHandler(api_key, api_secret_key)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def search_for_pod(search_txt, num_results):
    '''
    '''
    user_search = api.search_users((search_txt), count=num_results)
    search_results = []
    x = 1
    for user in user_search:
        search_results.append([search_txt, x, user.screen_name, user.id,
                               user.followers_count,
                               user.description.encode('utf-8')])
        x += 1
    return search_results


def get_all_followers(screen_name):
    '''
    '''
    followers_list = []
    print('Finding {} followers.'.format(screen_name), end='')
    for page in tweepy.Cursor(api.followers_ids, screen_name=screen_name,
                              monitor_rate_limit=True, wait_on_rate_limit=True,
                              wait_on_rate_limit_notify=True,
                              retry_count=5, retry_delay=5).pages(4):
        followers_list.extend(page)
        print('.', end='')
        time.sleep(0.7)
    print('.Done')
    return followers_list


def get_x_tweets(user_id, x):
    ''' Requests latest x tweets of user user_id.
        Returns List of full tweet text.
    '''
    alltweets = []
    new_tweets = api.user_timeline(user_id=user_id, count=x, tweet_mode='extended')
    for tweet in new_tweets:
        try:
            alltweets.append(tweet.retweeted_status.full_text)
        except AttributeError:  # Not a Retweet
            alltweets.append(tweet.full_text)
    return alltweets
