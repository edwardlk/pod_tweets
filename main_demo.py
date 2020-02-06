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

data_dir = './follower_ids/'
resources_dir = './resources/'

rank_matrix = pd.read_pickle(resources_dir + 'rank_matrix.pkl')

pod_list = rank_matrix.columns

pod_usr = pod_list[0]
pod_compare = pod_list[1]
pod_usr = st.sidebar.selectbox(
    'your podcast?',
    pod_list)

# pod_list.remove(pod_usr)
pod_compare = st.sidebar.selectbox(
    'comparison podcast?',
    pod_list)

rank_matrix[[pod_usr, pod_compare]]

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

st.write(bokeh_figure)
