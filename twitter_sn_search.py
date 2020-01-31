import requests
import pandas as pd
import time
from bs4 import BeautifulSoup


# Get links to podcast pages on podchaser
podchaser_search = 'https://www.podchaser.com/search/podcasts/q/'

pod_list_df = pd.read_csv('pod_list.csv', index_col=0)
podchaser_df = pd.read_csv('podchaser.csv')

for x1 in range(100, 500):
    start = time.time()
    temp_podchaser_df = pd.DataFrame(columns=['title', 'link', 'twitter_s_n'])

    search_text = pod_list_df.at[x1, 'Name']
    print('Searching "{}"...'.format(search_text))
    page = requests.get('https://www.podchaser.com/search/podcasts/q/' + search_text)

    soup = BeautifulSoup(page.content, 'html.parser')

    x2 = 0
    for link in soup.find_all('a', class_="entityLink_isn7nc-o_O-link_13xlah4"):
        temp_podchaser_df.at[x2, 'title'] = link.find('img').get('alt')
        temp_podchaser_df.at[x2, 'link'] = link.get('href')
        x2 += 1

    podchaser_df = podchaser_df.append(temp_podchaser_df, ignore_index=True)
    podchaser_df.to_csv('podchaser.csv', index=False)
    print('Done, waiting...')
    time.sleep(15)

podchaser_df = pd.read_csv('podchaser.csv', lineterminator='\n')

podchaser_df = podchaser_df.sort_values(by=['title'])
podchaser_df = podchaser_df.drop_duplicates()

podchaser_df.to_csv('podchaser1.csv', index=False)

# Search for twitter screen names on podchaser
# Need to correct finding creators multiple times
podchaser_df = pd.read_csv('podchaser1.csv', lineterminator='\n', dtype='object')
podchaser_df = pd.read_csv('get_these_SM_Jan30.csv')
link_start = 'https://www.podchaser.com'
twitter_url = 'twitter.com/'
print(podchaser_df.dtypes)

for x3 in range(len(podchaser_df)):
    print('{} search: {}...'.format(x3, podchaser_df.at[x3, 'Name']), end='')
    creator_list = []
    pod_link = link_start + podchaser_df.at[x3, 'link']
    pod_page = requests.get(pod_link, allow_redirects=False)
    pod_soup = BeautifulSoup(pod_page.content, 'html.parser')
    for link in pod_soup.find_all('span', class_="subtitle_1tdg67i"):
        if twitter_url in link.get('title'):
            podchaser_df.at[x3, 'twitter_s_n'] = [link.get('title')[12:]]
        else:
            print('none, looking for creators...')
            for link in pod_soup.find_all('a', class_="subtitleLink_1h2tj08"):
                if 'creators' in link.get('href'):
                    creator_page = requests.get('https://www.podchaser.com' + link.get('href'))
                    creator_soup = BeautifulSoup(creator_page.content, 'html.parser')
                    for link in creator_soup.find_all('a', class_="socialLinkText_7c78fw"):
                        if 'twitter.com' in link.get('href'):
                            creator_list.append(link.get('href')[24:])
                    time.sleep(5)
            podchaser_df.at[x3, 'twitter_s_n'] = creator_list
    print('found {}'.format(podchaser_df.at[x3, 'twitter_s_n']))
    time.sleep(1)
    if x3 % 20 == 0:
        podchaser_df.to_csv('podchaserJan30.csv', index=False)

podchaser_df.to_csv('podchaserJan30.csv', index=False)
