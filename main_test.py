import os
import time
import datetime
import tweepy
import numpy as np
import pandas as pd
from twitter_funcs import(get_x_tweets, get_all_followers, search_for_pod)


full_pod_list = ['BingeModeHarryPotter.csv',
                    'BitchSeshARealHousewivesBreakdown.csv',
                    'Caliphate.csv',
                    'CLUBLIFE.csv',
                    'CommonSensewithDanCarlin.csv',
                    'CongratulationswithChrisDElia.csv',
                    'CrimeJunkie.csv',
                    'CriticalRole.csv',
                    'Cults.csv',
                    'DanCarlinsHardcoreHistory.csv',
                    'DearSugars.csv',
                    'DeathSexMoney.csv',
                    'DirtyJohn.csv',
                    'DISGRACELAND.csv',
                    'DuncanTrussellFamilyHour.csv',
                    'EarHustle.csv',
                    'Embedded.csv',
                    'FiveThirtyEightPolitics.csv',
                    'ForePlay.csv',
                    'FreshAir.csv',
                    'GettingCuriouswithJonathanVanNess.csv',
                    'GiantBombcast.csv',
                    'GirlonGuywithAishaTyler.csv',
                    'GuysWeFd.csv',
                    'HappierwithGretchenRubin.csv',
                    'Heavyweight.csv',
                    'HelloFromTheMagicTavern.csv',
                    'HeresTheThingwithAlecBaldwin.csv',
                    'HiddenBrain.csv',
                    'Homecoming.csv',
                    'HowDidThisGetMade.csv',
                    'HowIBuiltThiswithGuyRaz.csv',
                    'IAMRAPAPORTSTEREOPODCAST.csv',
                    'ID10TwithChrisHardwick.csv',
                    'IfIWereYou.csv',
                    'InterceptedwithJeremyScahill.csv',
                    'IntheDark.csv',
                    'Invisibilia.csv',
                    'JalenJacoby.csv',
                    'JennaJulienPodcast.csv',
                    'JockoPodcast.csv',
                    'JudgeJohnHodgman.csv',
                    'JuicyScoopwithHeatherMcDonald.csv',
                    'LadyGang.csv',
                    'LastPodcastOnTheLeft.csv',
                    'LeVarBurtonReads.csv',
                    'Lore.csv',
                    'LouderWithCrowder.csv',
                    'Majority54.csv',
                    'MentalIllnessHappyHour.csv',
                    'MissingMurderedFindingCleo.csv',
                    'MSNBCRachelMaddowvideo.csv',
                    'MyBrotherMyBrotherAndMe.csv',
                    'MyDadWroteAPorno.csv',
                    'MyFavoriteMurderwithKarenKilgariffandGeorgiaHardstark.csv',
                    'MysteryShow.csv',
                    'NotTooDeepwithGraceHelbig.csv',
                    'OffTheVinewithKaitlynBristowe.csv',
                    'OnBeingwithKristaTippett.csv',
                    'OntheMedia.csv',
                    'OprahsSuperSoulConversations.csv',
                    'PardonMyTake.csv',
                    'PennsSundaySchool.csv',
                    'PodcastsTheMikeOMearaShow.csv',
                    'PodSavethePeople.csv',
                    'PoliticalGabfest.csv',
                    'PopCultureHappyHour.csv',
                    'Radiolab.csv',
                    'RadiolabPresentsMorePerfect.csv',
                    'RealCrimeProfile.csv',
                    'ReplyAll.csv',
                    'RevisionistHistory.csv',
                    'Revolutions.csv',
                    'RISETogetherPodcast.csv',
                    'RoosterTeethPodcast.csv',
                    'RoughTranslation.csv',
                    'SavageLovecast.csv',
                    'SawbonesAMaritalTourofMisguidedMedicine.csv',
                    'ShaneAndFriends.csv',
                    'SmallTownMurder.csv',
                    'SnapJudgment.csv',
                    'SomeoneKnowsSomething.csv',
                    'SomethingtoWrestlewithBrucePrichard.csv',
                    'SpittinChiclets.csv',
                    'SPONTANEANATIONwithPaulFTompkins.csv',
                    'StarTalkRadio.csv',
                    'StartUpPodcast.csv',
                    'StillProcessing.csv',
                    'STown.csv',
                    'StraightUpwithStassi.csv',
                    'Strangers.csv',
                    'StuffYouMissedinHistoryClass.csv',
                    'StuffYouShouldKnow.csv',
                    'SwordandScale.csv',
                    'TalkIsJericho.csv',
                    'TANIS.csv',
                    'TEDRadioHour.csv',
                    'TenMinutePodcast.csv',
                    'TerribleThanksForAsking.csv',
                    'TheAdamandDrDrewShow.csv',
                    'TheAdventureZone.csv',
                    'TheAndrewKlavanShow.csv',
                    'TheArtofManliness.csv',
                    'TheAxeFileswithDavidAxelrod.csv',
                    'TheBenandAshleyIAlmostFamousPodcast.csv',
                    'TheBenShapiroShow.csv',
                    'TheBillSimmonsPodcast.csv',
                    'TheBrilliantIdiots.csv',
                    'TheChaleneShowDietFitnessLifeBalance.csv',
                    'TheDaily.csv',
                    'TheDanPatrickShowonPodcastOne.csv',
                    'TheDavePortnoyShow.csv',
                    'TheDollopwithDaveAnthonyandGarethReynolds.csv',
                    'TheEzraKleinShow.csv',
                    'TheFighterTheKid.csv',
                    'TheGaryVeeAudioExperience.csv',
                    'TheGlennBeckProgram.csv',
                    'TheHappyHourwithJamieIvey.csv',
                    'TheHerdwithColinCowherd.csv',
                    'TheHistoryofRome.csv',
                    'TheJoeBuddenPodcastwithRoryMal.csv',
                    'TheJoeRoganExperience.csv',
                    'TheJordanBPetersonPodcast.csv',
                    'TheLongestShortestTime.csv',
                    'thememorypalace.csv',
                    'TheMinimalistsPodcast.csv',
                    'TheMoth.csv',
                    'TheNoSleepPodcast.csv',
                    'ThePatMcAfeeShow.csv',
                    'TheRachelMaddowShow.csv',
                    'TheRead.csv',
                    'TheRichRollPodcast.csv',
                    'TheRossBolenPodcast.csv',
                    'TheRussilloShow.csv',
                    'TheSkepticsGuidetotheUniverse.csv',
                    'TheSkinnyConfidentialHimHerPodcast.csv',
                    'TheSteveAustinShow.csv',
                    'TheTaiLopezShow.csv',
                    'TheTimFerrissShow.csv',
                    'TheUnderdog.csv',
                    'TheVanishedPodcast.csv',
                    'TheWeeklyPlanet.csv',
                    'TheWestWingWeekly.csv',
                    'ThisAmericanLife.csv',
                    'ThrowingShade.csv',
                    'TigerBelly.csv',
                    'TimesuckwithDanCummins.csv',
                    'Trumpcast.csv',
                    'TruthJusticewithBobRuff.csv',
                    'Undisclosed.csv',
                    'UpandVanished.csv',
                    'UpFirst.csv',
                    'WatchWhatCrappens.csv',
                    'WelcometoNightVale.csv',
                    'WithFriendsLikeThese.csv',
                    'WorkLifewithAdamGrant.csv',
                    'YouMadeItWeirdwithPeteHolmes.csv',
                    'YouMustRememberThis.csv',
                    'YoungHouseLoveHasAPodcast.csv',
                    'YourMomsHousewithChristinaPandTomSegura.csv']

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
# resource_dir = '/home/ed/github/pod_tweets/'
# data_file = 'podchaserJan30fixed.csv'
# pod_screenName = pd.read_csv(resource_dir + data_file, index_col=0)
#
# idx_list = pod_screenName.index.tolist()
#
# for x in idx_list[180:]:
#     if not pd.isnull(pod_screenName.loc[x, 'twitter_s_n']):
#         temp_df = pd.DataFrame(get_all_followers(
#                      pod_screenName.at[x, 'twitter_s_n']), columns=['user_id'])
#         fname = ''.join(e for e in pod_screenName.at[x, 'Name'] if e.isalnum())
#         temp_df.to_csv('follower_ids/{}.csv'.format(fname), index=False)
#         print('{} files done'.format(x))

# Create df of podcast followers' latest tweets
start_time = time.time()
pod_follower_list = ['ArtofWrestling.csv', 'AstonishingLegends.csv',
                     'AtlantaMonster.csv', 'Bertcastspodcast.csv']

for pdcst in full_pod_list:
    follow_ids = 'follower_ids/' + pdcst
    follow_twts = 'follower_twts/' + pdcst

    temp_df = pd.read_csv(follow_ids)
    temp_df['tweets'] = [[]]*len(temp_df)

    users_to_scrape = 4000
    if len(temp_df) < 4000:
        users_to_scrape = len(temp_df)

    print('Gathering {} followers...'.format(pdcst[:-4]))
    for x in range(users_to_scrape):
        try:
            user_tweet_list = get_x_tweets(temp_df.at[x, 'user_id'], 10)
        except tweepy.TweepError:
            user_tweet_list = []
            print("Failed to run the command on that user, Skipping...")
        temp_df.at[x, 'tweets'] = user_tweet_list

        if x % 10 == 0:
            print('{} users done'.format(x))
        if x % 100 == 0:
            temp_df.to_csv(follow_twts, index=False)
        time.sleep(0.700)

    temp_df.to_csv(follow_twts, index=False)

    total_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('It took {}.'.format(total_time))
