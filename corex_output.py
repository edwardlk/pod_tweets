import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity


def convert_all_df(all_df):
    listCol = all_df.columns.tolist()
    for x in ['text', 'emojis', 'hashtags', 'mentions']:
        listCol.remove(x)

    all_df = all_df[listCol]

    all_df_grouped = all_df.groupby(['podcast_name']).mean()

    return all_df_grouped


def precision_at_k(df, k):
    num = 0
    denom = df[:k].sum()
    for x in range(k):
        if df.loc[x] == 1:
            num += df[:x+1].sum() / (x+1)
    if denom == 0:
        return np.nan
    return num/denom


def shuffle_df(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


resource_dir = './resources/'
corex_dir = './corex_models/trained-topic-model/'

# Get topic model words dict:
topic_txt = pd.read_csv(corex_dir + 'topics.txt', delimiter=':', index_col=0, header=None)
topic_dict = {}
for x in range(len(topic_txt)):
    key_str = 'topic_{}'.format(x)
    val_str = topic_txt.iloc[x, 0].split(',')[0]
    topic_dict[key_str] = val_str

#
all_df = pd.read_csv(resource_dir + 'all_df.csv', index_col=0)

all_df_grouped = convert_all_df(all_df)

listCol = all_df.columns.tolist()

for x in ['text', 'emojis', 'hashtags', 'mentions', 'user_id', 'podcast_name']:
    listCol.remove(x)

v = 0.002
sel = VarianceThreshold(threshold=(v*(1-v)))
# print(all_df_grouped[listCol].shape)
X_out = sel.fit_transform(all_df_grouped[listCol])
# print(X_out.shape)

topicTF = sel.get_support()

good_topics = []
for x in range(len(topicTF)):
    if topicTF[x]:
        good_topics.append(listCol[x])

good_topic_dict = {}
for x in good_topics:
    good_topic_dict[x] = topic_dict[x]

good_topics_dict = {
    'topic_1': 'win', 'topic_4': 'kobe bryant', 'topic_6': 'super bowl',
    'topic_10': 'mental health', 'topic_13': 'religion',
    'topic_15': 'social justice', 'topic_16': 'brexit',
    'topic_19': 'social justice',
    'topic_20': 'music',
    'topic_21': 'movie',
    'topic_22': 'food',
    'topic_24': 'space',
    'topic_33': 'thought',
    'topic_34': 'point',
    'topic_38': 'come',
    'topic_42': 'adoption fees',
    'topic_49': 'tech',
    'topic_50': 'going strong',
    'topic_52': 'friends',
    'topic_57': 'existential',
    'topic_60': 'german',
    'topic_64': 'profanity',
    'topic_65': 'david bowie',
    'topic_69': 'stood injustice',
    'topic_73': 'just started day',
    'topic_77': 'past years',
    'topic_91': 'gaming',
    'topic_92': 'hard time',
    'topic_97': 'family members'}

final_cols = ['podcast_name']
final_cols.extend(good_topics)

sim_df = all_df_grouped[good_topics]

simdf_col = sim_df.columns.tolist()
new_simdf_col = []
for x in simdf_col:
    new_simdf_col.append(good_topics_dict[x])

temp_sim_df = sim_df
temp_sim_df.columns = new_simdf_col

temp_sim_df.to_csv('topic_percent_matrix223.csv')

rank_matrix = pd.DataFrame()
for x3 in range(len(sim_df)):
    df = pd.DataFrame(sim_df.iloc[x3])
    df = df.sort_values(by=[str(df.columns[0])], ascending=False)
    rank_matrix[df.columns[0]] = df.index.tolist()
rank_matrix.to_csv('list_of_topics223.csv')

# Cosine Similarity
cos_sim = cosine_similarity(sim_df)
cos_sim_df = pd.DataFrame(cos_sim, columns=sim_df.index)
cos_sim_df = cos_sim_df.T
cos_sim_df.columns = cos_sim_df.index.tolist()

cos_sim_df.to_csv('cos_sim_df223.csv')

sim_dict = {}
for pod in cos_sim_df.index.tolist():
    sim_dict[pod] = cos_sim_df[pod].sort_values(ascending=False).index.tolist()[1:]

rec_rank_matrix = pd.DataFrame(sim_dict)

search_dict = joblib.load(resource_dir + 'rec-ment_dict.joblib')

for x in rec_rank_matrix.columns.tolist():
    for y in range(len(rec_rank_matrix)):
        if x in search_dict.keys():
            if rec_rank_matrix.loc[y, x] in search_dict[x]:
                rec_rank_matrix.loc[y, x] = 1
            else:
                rec_rank_matrix.loc[y, x] = np.nan
        else:
            rec_rank_matrix.loc[y, x] = np.nan

rec_rank_matrix.to_csv('rec_rank_test.cav')

# precision at k
precision_all_df = pd.DataFrame()

k_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
k_cols = rec_rank_matrix.columns.tolist()

precision_k_df = pd.DataFrame(index=k_index, columns=k_cols)

for x in k_cols:
    for y in k_index:
        precision_k_df.loc[y, x] = precision_at_k(rec_rank_matrix[x], y)

precision_k_df.to_csv('precision_k_test.csv')
precision_all_df['model'] = precision_k_df.mean(axis=1)

# randomize recommendations:
for n in range(20):
    rand_rec_rank_matrix = shuffle_df(rec_rank_matrix)
    rand_precision_k_df = pd.DataFrame(index=k_index, columns=k_cols)

    for x in k_cols:
        for y in k_index:
            rand_precision_k_df.loc[y, x] = precision_at_k(rand_rec_rank_matrix[x], y)

    rand_precision_k_df.to_csv('rand_precision_k_test.csv')
    precision_all_df['rand_{}'.format(n)] = rand_precision_k_df.mean(axis=1)

precision_all_df.to_csv('all_precision_k_test.csv')
