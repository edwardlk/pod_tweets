import os

# import pickle
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
# import scipy.sparse as ss
import numpy as np
# from sklearn.decomposition import TruncatedSVD, PCA
# from sklearn.preprocessing import normalize
# from sklearn.base import BaseEstimator
# from sklearn.utils import check_array
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.manifold import TSNE
# from os.path import isfile
# import subprocess
import streamlit as st


def calc_common_usrs(data_dir, pod1, pod2):
    pod_list1 = [line.rstrip('\n') for line in open(data_dir + pod1)]
    pod_list1.pop(0)
    pod_list2 = [line.rstrip('\n') for line in open(data_dir + pod2)]
    pod_list2.pop(0)
    num_same = set(pod_list1) & set(pod_list2)
    return len(num_same), len(pod_list1), len(pod_list2)


def return_comm_users(pod_common_usrs, pod_2, pod_1):
    df = pod_common_usrs[pod_common_usrs['podcast_1'] == (pod_1 + '.csv')]
    df = df[df['podcast_2'] == (pod_2 + '.csv')]
    X = df['comm_users'].tolist()

    if not X:
        df = pod_common_usrs[pod_common_usrs['podcast_1'] == (str(pod_2) + '.csv')]
        df = df[df['podcast_2'] == (str(pod_1) + '.csv')]
        X = df['comm_users'].tolist()

    return X[0]


pod_pop_arr = ['60Minutes.csv', '48Hours.csv', '1A.csv', '99Invisible.csv',
               'BehindtheBastards.csv', '83WeekswithEricBischoff.csv',
               'AccidentalTechPodcast.csv', 'Radiolab.csv', 'SmallTownMurder.csv',
                'GirlonGuywithAishaTyler.csv', 'HelloFromTheMagicTavern.csv',
                'YouMustRememberThis.csv', 'RoosterTeethPodcast.csv',
                'WelcometoNightVale.csv', 'SPONTANEANATIONwithPaulFTompkins.csv',
                'GettingCuriouswithJonathanVanNess.csv', 'TheWeeklyPlanet.csv',
                'TheBrilliantIdiots.csv', 'OnBeingwithKristaTippett.csv',
                'SawbonesAMaritalTourofMisguidedMedicine.csv',
                'AtlantaMonster.csv', 'StraightUpwithStassi.csv',
                'PardonMyTake.csv', 'GuysWeFd.csv', 'TheRachelMaddowShow.csv',
                'TheJoeBuddenPodcastwithRoryMal.csv', 'TheHistoryofRome.csv',
                'TheAdventureZone.csv', 'RevisionistHistory.csv', 'TheRead.csv',
                'HowIBuiltThiswithGuyRaz.csv',
                'SomethingtoWrestlewithBrucePrichard.csv', 'IntheDark.csv',
                'AceOnTheHouse.csv', 'PennsSundaySchool.csv', 'ArtofWrestling.csv',
                'StarTalkRadio.csv', 'YourMomsHousewithChristinaPandTomSegura.csv',
                'HeresTheThingwithAlecBaldwin.csv', 'TheAdamandDrDrewShow.csv',
                'STown.csv', 'MyDadWroteAPorno.csv', 'TheEzraKleinShow.csv',
                'Majority54.csv', 'MissingMurderedFindingCleo.csv',
                'TEDRadioHour.csv', 'WorkLifewithAdamGrant.csv', 'LadyGang.csv',
                'IAMRAPAPORTSTEREOPODCAST.csv', 'TheAndrewKlavanShow.csv',
                'JockoPodcast.csv', 'NotTooDeepwithGraceHelbig.csv',
                'AlisonRosenIsYourNewBestFriend.csv',
                'BitchSeshARealHousewivesBreakdown.csv', 'LouderWithCrowder.csv',
                'Homecoming.csv', 'SavageLovecast.csv', 'TheRichRollPodcast.csv',
                'AndThatsWhyWeDrink.csv', 'RealCrimeProfile.csv',
                '30For30Podcasts.csv', 'ArmchairExpertwithDaxShepard.csv',
                'TheRossBolenPodcast.csv', 'OprahsSuperSoulConversations.csv',
                'ReplyAll.csv', 'GiantBombcast.csv', 'Bertcastspodcast.csv',
                'TheSteveAustinShow.csv', 'PodSaveAmerica.csv',
                'TheHerdwithColinCowherd.csv', 'Caliphate.csv',
                'TheDanPatrickShowonPodcastOne.csv', 'TerribleThanksForAsking.csv',
                'TheFighterTheKid.csv', 'DuncanTrussellFamilyHour.csv',
                'Invisibilia.csv', 'TheTaiLopezShow.csv',
                'JuicyScoopwithHeatherMcDonald.csv', '2DopeQueens.csv',
                'HappierwithGretchenRubin.csv', 'TimesuckwithDanCummins.csv',
                'TheRussilloShow.csv', 'WithFriendsLikeThese.csv',
                'TheJoeRoganExperience.csv', 'Lore.csv', 'JalenJacoby.csv',
                'TheDaily.csv', 'StillProcessing.csv', 'PoliticalGabfest.csv',
                'LeVarBurtonReads.csv', 'AdamCarollaShow.csv',
                'MyFavoriteMurderwithKarenKilgariffandGeorgiaHardstark.csv',
                'JudgeJohnHodgman.csv', 'ThisAmericanLife.csv',
                'SpittinChiclets.csv', 'YouMadeItWeirdwithPeteHolmes.csv',
                'TenMinutePodcast.csv', 'Trumpcast.csv', 'TigerBelly.csv',
                'RadiolabPresentsMorePerfect.csv',
                'TheChaleneShowDietFitnessLifeBalance.csv',
                'TheBillSimmonsPodcast.csv', 'CriticalRole.csv', 'CLUBLIFE.csv',
                'MysteryShow.csv', 'OffTheVinewithKaitlynBristowe.csv',
                'JennaJulienPodcast.csv', 'TheMinimalistsPodcast.csv',
                'TheGaryVeeAudioExperience.csv', 'TheGlennBeckProgram.csv',
                'MSNBCRachelMaddowvideo.csv', 'TheWestWingWeekly.csv',
                'TheDavePortnoyShow.csv', 'TheJordanBPetersonPodcast.csv',
                'TheBenandAshleyIAlmostFamousPodcast.csv', 'Heavyweight.csv',
                'Undisclosed.csv', 'AnnaFarisIsUnqualified.csv',
                'FiveThirtyEightPolitics.csv', 'FreshAir.csv', 'ThrowingShade.csv',
                'LastPodcastOnTheLeft.csv', 'ID10TwithChrisHardwick.csv',
                'OntheMedia.csv', 'BingeModeHarryPotter.csv',
                'MyBrotherMyBrotherAndMe.csv', 'TheAxeFileswithDavidAxelrod.csv',
                'UpandVanished.csv', 'CommonSensewithDanCarlin.csv',
                'CongratulationswithChrisDElia.csv', 'ThePatMcAfeeShow.csv',
                'TheTimFerrissShow.csv', 'ForePlay.csv', 'ShaneAndFriends.csv',
                'TheBenShapiroShow.csv', 'DeathSexMoney.csv',
                'TheSkinnyConfidentialHimHerPodcast.csv',
                'DanCarlinsHardcoreHistory.csv', 'PodSavethePeople.csv',
                'Revolutions.csv', 'HowDidThisGetMade.csv', 'TalkIsJericho.csv',
                'TheDollopwithDaveAnthonyandGarethReynolds.csv',
                'TruthJusticewithBobRuff.csv', 'TheVanishedPodcast.csv',
                'thememorypalace.csv', 'CrimeJunkie.csv',
                'PopCultureHappyHour.csv', 'SnapJudgment.csv', 'SwordandScale.csv',
                'WatchWhatCrappens.csv', 'AstonishingLegends.csv', 'UpFirst.csv',
                'TheHappyHourwithJamieIvey.csv', 'AliceIsntDead.csv',
                'Strangers.csv', 'TANIS.csv', 'StartUpPodcast.csv',
                'TheArtofManliness.csv', 'TheNoSleepPodcast.csv', 'DearSugars.csv',
                'StuffYouShouldKnow.csv', 'InterceptedwithJeremyScahill.csv',
                '1YearDailyAudioBible.csv', 'RISETogetherPodcast.csv',
                'EarHustle.csv', 'SeincastASeinfeldPodcast.csv',
                'PodcastsTheMikeOMearaShow.csv',
                'StuffYouMissedinHistoryClass.csv', 'SomeoneKnowsSomething.csv',
                'PlanetMoney.csv', 'Embedded.csv', 'KnowledgeFight.csv',
                'HiddenBrain.csv', 'TheSkepticsGuidetotheUniverse.csv',
                'DirtyJohn.csv', 'RoughTranslation.csv',
                'AmericanHistoryTellers.csv', 'Accused.csv',
                'TheLongestShortestTime.csv', 'DISGRACELAND.csv',
                'TheUnderdog.csv', 'TheMoth.csv', 'YoungHouseLoveHasAPodcast.csv',
                'Cults.csv', 'IfIWereYou.csv', 'MentalIllnessHappyHour.csv']

fllw_cnt = {'OprahsSuperSoulConversations': 42852066, 'MSNBCRachelMaddowvideo': 9868196, 'TheRachelMaddowShow': 9868196, 'ShaneAndFriends': 9732900, 'TheBillSimmonsPodcast': 5861576, 'TheJoeRoganExperience': 5764309, 'CLUBLIFE': 5019775, 'TheSteveAustinShow': 4634639, 'TalkIsJericho': 3589921, 'TheEzraKleinShow': 2559225, 'TheBenShapiroShow': 2531427, 'TheGaryVeeAudioExperience': 2118223, 'LeVarBurtonReads': 1878729, 'PennsSundaySchool': 1819319, 'JalenJacoby': 1787513, 'ThePatMcAfeeShow': 1733874, 'TheTimFerrissShow': 1642970, 'TheHerdwithColinCowherd': 1473885, 'TheJordanBPetersonPodcast': 1398570, 'NotTooDeepwithGraceHelbig': 1281812, 'TheGlennBeckProgram': 1264010, 'SPONTANEANATIONwithPaulFTompkins': 1249653, 'TheAxeFileswithDavidAxelrod': 1196490, 'TheDavePortnoyShow': 1188861, 'ArmchairExpertwithDaxShepard': 1125492, 'CongratulationswithChrisDElia': 1066960, 'TheJoeBuddenPodcastwithRoryMal': 1045202, 'PodSavethePeople': 1034231, 'LouderWithCrowder': 988334, 'GettingCuriouswithJonathanVanNess': 819646, 'TheFighterTheKid': 779731, 'PardonMyTake': 716940, 'JennaJulienPodcast': 701095, 'TheTaiLopezShow': 698959, 'RevisionistHistory': 637712, 'CriticalRole': 576196, 'Bertcastspodcast': 553802, 'TheAdamandDrDrewShow': 530417, 'AceOnTheHouse': 530417, 'AnnaFarisIsUnqualified': 495297, 'GirlonGuywithAishaTyler': 494297, 'TheRussilloShow': 489211, 'PodSaveAmerica': 480175, 'OffTheVinewithKaitlynBristowe': 456363, 'StraightUpwithStassi': 448565, 'RoosterTeethPodcast': 435846, 'TheDanPatrickShowonPodcastOne': 434861, 'YouMadeItWeirdwithPeteHolmes': 431563, 'MyFavoriteMurderwithKarenKilgariffandGeorgiaHardstark': 425423, 'IAMRAPAPORTSTEREOPODCAST': 417446, '30For30Podcasts': 393754, 'JockoPodcast': 390419, 'Caliphate': 389576, 'PlanetMoney': 360348, 'SavageLovecast': 355360, 'Majority54': 337683, 'MyBrotherMyBrotherAndMe': 316000, 'RadiolabPresentsMorePerfect': 311137, 'WorkLifewithAdamGrant': 294645, 'DuncanTrussellFamilyHour': 285679, 'SpittinChiclets': 283082, 'HeresTheThingwithAlecBaldwin': 279385, 'FreshAir': 278914, 'ThisAmericanLife': 268314, 'JuicyScoopwithHeatherMcDonald': 267024, 'TheAdventureZone': 256404, 'CommonSensewithDanCarlin': 253749, 'DanCarlinsHardcoreHistory': 253749, 'ArtofWrestling': 253535, 'BingeModeHarryPotter': 225267, 'TheBenandAshleyIAlmostFamousPodcast': 224906, 'TheDaily': 212621, 'TheRead': 205695, 'ForePlay': 195582, 'GiantBombcast': 166770, 'TheAndrewKlavanShow': 166093, 'TenMinutePodcast': 163934, 'YourMomsHousewithChristinaPandTomSegura': 159041, 'TheBrilliantIdiots': 150907, 'HappierwithGretchenRubin': 144528, '48Hours': 138381, 'YouMustRememberThis': 125112, 'Trumpcast': 124863, 'HowDidThisGetMade': 122040, 'ID10TwithChrisHardwick': 108028, 'BitchSeshARealHousewivesBreakdown': 107171, 'ReplyAll': 105815, 'TheChaleneShowDietFitnessLifeBalance': 101482, 'Homecoming': 94587, 'LadyGang': 93428, 'TheRichRollPodcast': 90097, 'SawbonesAMaritalTourofMisguidedMedicine': 89505, '2DopeQueens': 84296, 'StillProcessing': 83934, '83WeekswithEricBischoff': 83161, 'SomethingtoWrestlewithBrucePrichard': 83161, 'Undisclosed': 82453, 'AlisonRosenIsYourNewBestFriend': 77924, 'WelcometoNightVale': 76179, 'TEDRadioHour': 67905, 'HowIBuiltThiswithGuyRaz': 67905, 'LastPodcastOnTheLeft': 63048, 'WithFriendsLikeThese': 58849, '1A': 58727, 'TheWestWingWeekly': 58159, '99Invisible': 57109, 'OnBeingwithKristaTippett': 56803, 'Lore': 55815, 'JudgeJohnHodgman': 55443, 'GuysWeFd': 53496, 'PoliticalGabfest': 50825, 'TheMinimalistsPodcast': 50587, 'MysteryShow': 48928, 'TheHistoryofRome': 48706, 'Revolutions': 48705, 'AdamCarollaShow': 43769, 'TheDollopwithDaveAnthonyandGarethReynolds': 43716, 'TheWeeklyPlanet': 42687, 'MyDadWroteAPorno': 41293, 'StarTalkRadio': 39330, '60Minutes': 39038, 'UpandVanished': 38737, 'AtlantaMonster': 38737, 'TheRossBolenPodcast': 38574, 'Invisibilia': 37072, 'ThrowingShade': 36386, 'STown': 34374, 'Heavyweight': 33748, 'MissingMurderedFindingCleo': 33623, 'BehindtheBastards': 31512, 'TerribleThanksForAsking': 31250, 'AndThatsWhyWeDrink': 30190, 'AccidentalTechPodcast': 29116, 'DeathSexMoney': 28170, 'TheSkinnyConfidentialHimHerPodcast': 26791, 'HelloFromTheMagicTavern': 26362, 'Radiolab': 24857, 'OntheMedia': 23828, 'TimesuckwithDanCummins': 23422, 'FiveThirtyEightPolitics': 23065, 'RealCrimeProfile': 22892, 'SmallTownMurder': 22497, 'TigerBelly': 22493, 'IntheDark': 22224, 'TruthJusticewithBobRuff': 19602, 'AliceIsntDead': 18390, 'TheVanishedPodcast': 17962, 'CrimeJunkie': 17523, 'thememorypalace': 17495, 'PopCultureHappyHour': 16624, 'SnapJudgment': 15058, 'SwordandScale': 14858, 'WatchWhatCrappens': 14775, 'AstonishingLegends': 13341, 'UpFirst': 13127, 'TheHappyHourwithJamieIvey': 11437, 'Strangers': 11308, 'TANIS': 10217, 'StartUpPodcast': 10148, 'TheArtofManliness': 9784, 'TheNoSleepPodcast': 9401, 'DearSugars': 8895, 'StuffYouShouldKnow': 7917, 'RISETogetherPodcast': 7311, 'InterceptedwithJeremyScahill': 7093, '1YearDailyAudioBible': 6586, 'EarHustle': 6077, 'SeincastaSeinfeldPodcast': 5752, 'PodcastsTheMikeOMearaShow': 5613, 'StuffYouMissedinHistoryClass': 5227, 'SomeoneKnowsSomething': 5148, 'Embedded': 4536, 'KnowledgeFight': 4334, 'HiddenBrain': 4111, 'TheSkepticsGuidetotheUniverse': 3877, 'DirtyJohn': 3760, 'RoughTranslation': 3760, 'AmericanHistoryTellers': 3513, 'Accused': 3173, 'TheLongestShortestTime': 2503, 'DISGRACELAND': 2028, 'TheUnderdog': 1288, 'TheMoth': 760, 'YoungHouseLoveHasAPodcast': 718, 'Cults': 332, 'IfIWereYou': 266, 'MentalIllnessHappyHour': 4}

data_dir = './follower_ids/'
resources_dir = './resources/'

cos_sim_pkl = 'rank_matrix_20200213.pkl'
topic_csv = 'corex-topics.txt'

# cos_sim = pd.read_pickle(resources_dir + cos_sim_pkl)
cos_sim = pd.read_csv(resources_dir + 'cos_sim_df.csv', index_col=0)
# rank_matrix = pd.read_pickle(resources_dir + 'rank_matrix_202002121312.pkl')
rank_matrix = pd.read_csv(resources_dir + 'list_of_topics.csv', index_col=0)
topic_txt_df = pd.read_csv(resources_dir + topic_csv, header=None, delimiter=':')
pod_common_users = pd.read_csv(resources_dir + 'pod_common_users.csv', index_col=0)
top_percent_df = pd.read_csv(resources_dir + 'topic_percent_matrix.csv', index_col=0)

# rank_matrix = pd.DataFrame()
# for x2 in range(len(sim_df)):
#     df = pd.DataFrame(sim_df.iloc[x2])
#     df = df.sort_values(by=[str(df.columns[0])], ascending=False)
#     rank_matrix[df.columns[0]] = df.index.tolist()

top_percent_df = top_percent_df.T

pod_list = rank_matrix.columns

topic_list = rank_matrix['1A'].sort_values().tolist()

pod_usr = pod_list[0]
# pod_compare = pod_list[1]
pod_usr = st.sidebar.selectbox(
    'Select your podcast',
    pod_list, index=65)

num_compare = st.sidebar.slider(
    'Select num most similar podcasts to compare:',
    min_value=1, max_value=10, value=3)

topics_view = st.sidebar.slider(
    'Select num topics to view:',
    min_value=1, max_value=10, value=5)

sim_list = cos_sim[pod_usr].sort_values(ascending=False).index.tolist()

st.write('The top 3 podcasts with audiences most similar to your own are {}, {}, and {}'.format(sim_list[1], sim_list[2], sim_list[3]))

options = st.sidebar.multiselect(
    'What podcasts do you want to compare?',
    sim_list, default=sim_list[1:num_compare+1])

other_topics = st.sidebar.multiselect(
    'What other topics do you want to explore?',
    topic_list)

pod_look = [pod_usr]
pod_look.extend(options)

rank_matrix[:topics_view][pod_look]

top_usr_topics = rank_matrix[:topics_view][pod_usr].tolist()

pod_usr_followers = fllw_cnt[pod_usr]

out_str = ('This % of your users also follow these podcasts: \n\n')
for x in pod_look[1:]:
    temp_str = '\t\t {}: {:.1f}% \n\n'.format(x, 100*return_comm_users(pod_common_users, pod_usr, x)/pod_usr_followers)
    out_str = out_str + temp_str

num_com = return_comm_users(pod_common_users, pod_usr, sim_list[1])

comm_usr_str = 'common user % = {}'.format(
    return_comm_users(pod_common_users, pod_usr, sim_list[1]))

st.write(out_str)

if other_topics != []:
    top_usr_topics.extend(other_topics)

# if not other_topics:
#     top_usr_topics = top_usr_topics.append(other_topics)

outdfdf = top_percent_df.loc[top_usr_topics, pod_look].T*100
st.write('What % of these audiences are talking about these topics:')
st.dataframe(outdfdf.apply(lambda x: round(x, 1)))


usr_topic_0 = rank_matrix.loc[0, pod_usr]
usr_text_0 = topic_txt_df.loc[int(usr_topic_0[6:]), 1]
usr_topic_1 = rank_matrix.loc[1, pod_usr]
usr_text_1 = topic_txt_df.loc[int(usr_topic_1[6:]), 1]
usr_topic_2 = rank_matrix.loc[2, pod_usr]
usr_text_2 = topic_txt_df.loc[int(usr_topic_2[6:]), 1]

output_txt = ('The topics your audience are most interested in are: \n\n'
              + '{}: {} \n\n'.format(usr_topic_0, usr_text_0)
              + '{}: {} \n\n'.format(usr_topic_1, usr_text_1)
              + '{}: {}'.format(usr_topic_2, usr_text_2))

# st.write(topic_txt_df)

# st.write(output_txt)
#
# if len(other_topics) > 0:
#     for x in other_topics:
#         st.write('{}: {} \n\n'.format(x, topic_txt_df.loc[int(x[6:]), 1]))

# pod_list = os.listdir(data_dir)
# num_pods = len(pod_list)

# pod_common_usrs = pd.DataFrame(columns=['podcast_1', 'podcast_2', 'comm_users'])
# pod_popularity = pd.DataFrame(pod_list, columns=['podcast'])
# pod_popularity['followers'] = 0
# pod_popularity = pod_popularity.set_index('podcast')
#
# for x1 in range(num_pods):
#     for x2 in range(x1+1, num_pods):
#         comm_num, pod_x1, pod_x2 = calc_common_usrs(data_dir, pod_list[x1], pod_list[x2])
#         pod_common_usrs.loc[len(pod_common_usrs)] = [pod_list[x1], pod_list[x2], comm_num]
#         pod_popularity.at[pod_list[x1], 'followers'] = pod_x1
#         pod_popularity.at[pod_list[x2], 'followers'] = pod_x2
# print(pod_popularity)
#
# pod_popularity = pod_popularity.sort_values(by='followers', ascending=False)
# pod_pop_arr = np.array(pod_popularity.index)
#
# index_map = dict(np.vstack([pod_pop_arr, np.arange(pod_pop_arr.shape[0])]).T)
#
# count_matrix = ss.coo_matrix((pod_common_usrs.comm_users,
#                               (pod_common_usrs.podcast_2.map(index_map),
#                                pod_common_usrs.podcast_1.map(index_map))),
#                              shape=(pod_pop_arr.shape[0], pod_pop_arr.shape[0]),
#                              dtype=np.float64)
#
# conditional_prob_matrix = count_matrix.tocsr()
# conditional_prob_matrix = normalize(conditional_prob_matrix, norm='l1', copy=False)
#
# reduced_data = TruncatedSVD(n_components=2).fit_transform(conditional_prob_matrix)
data_dump = resources_dir + '20193101_data.joblib'
reduced_data = load(data_dump)

cluster_pick = 3
kmeans = KMeans(init='k-means++', n_clusters=cluster_pick, n_init=10)
kmeans.fit(reduced_data)

# model_dump = resources_dir + '20193101_kmeans.joblib'
# kmeans = load(model_dump)

# Plot
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on PCA-reduced data\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# plt.show()
# st.pyplot()

pod_space_df = pd.DataFrame(reduced_data, columns=('x', 'y'))
pod_space_df['pod_name'] = pod_pop_arr[:31]

import hdbscan

clusterer = hdbscan.HDBSCAN(min_samples=5, metric='manhattan',
                            min_cluster_size=20).fit(reduced_data)
cluster_ids = clusterer.labels_
print('num clusters = {}'.format(clusterer.labels_.max()+1))

pod_space_df['cluster'] = cluster_ids

hbds_dump = resources_dir + '20190402_hdbscan_data.joblib'
pod_space_df = load(hbds_dump)


from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, value
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import plasma
from collections import OrderedDict

output_notebook()

# Construct a color palette and map clusters to colors
# palette = ['#777777'] + plasma(cluster_ids.max())
# colormap = LinearColorMapper(palette=palette, low=-1, high=cluster_ids.max())
# color_dict = {'field': 'cluster', 'transform': colormap}

# Set fill alpha globally
pod_space_df['fill_alpha'] = np.exp((reduced_data.min() -
                                     reduced_data.max()) / 5.0) + 0.05

# Build a column data source
plot_data = ColumnDataSource(pod_space_df)

# Custom callback for alpha adjustment
jscode="""
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    alpha = data['fill_alpha']
    for (i = 0; i < alpha.length; i++) {
         alpha[i] = Math.exp((start - end) / 5.0) + 0.05;
    }
    source.trigger('change');
"""

# Create the figure and add tools
bokeh_figure = figure(title='A Map of PodSpace',
                   plot_width = 700,
                   plot_height = 700,
                   tools=('pan, wheel_zoom, box_zoom, box_select, reset'),
                   active_scroll=u'wheel_zoom')

bokeh_figure.add_tools( HoverTool(tooltips = OrderedDict([('pod_name', '@pod_name'),
                                                       ('cluster', '@cluster')])))

# draw the subreddits as circles on the plot
bokeh_figure.circle(u'x', u'y', source=plot_data,
                 line_color=None, fill_alpha='fill_alpha',
                 size=10, hover_line_color=u'black')

# bokeh_figure.x_range.callback = CustomJS(args=dict(source=plot_data), code=jscode)
# bokeh_figure.y_range.callback = CustomJS(args=dict(source=plot_data), code=jscode)

# configure visual elements of the plot
bokeh_figure.title.text_font_size = value('18pt')
bokeh_figure.title.align = 'center'
bokeh_figure.xaxis.visible = False
bokeh_figure.yaxis.visible = False
bokeh_figure.grid.grid_line_color = None
bokeh_figure.outline_line_color = '#222222'

# st.write(bokeh_figure)
