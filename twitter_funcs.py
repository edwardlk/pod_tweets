import tweepy
import time

#Twitter API credentials
from auth import (
    api_key,
    api_secret_key,
    access_token,
    access_token_secret)

# Auths
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def get_all_followers(screen_name):
    followers_list = []
    for page in tweepy.Cursor(api.followers_ids, screen_name=screen_name).pages():
        followers_list.extend(page)
        time.sleep(5)
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
