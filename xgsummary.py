import pandas as pd
import numpy as np
import scipy.cluster as spc
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib as mpl
from string import *
import argparse, os, sys, joblib


### I/O ###

forest = joblib.load('fearfactorxg.joblib')
importances = forest.feature_importances_

def xG(s, xC, yC, p, sinp, ssle, z, st, pe, pxC, pyC):
    return importances[0]*s + importances[1]*xC + importances[2]*yC + importances[3]*p + importances[4]*sinp + importances[5]*ssle + importances[6]*z + importances[7]*st + importances[8]*pe + importances[9]*pxC + importances[10]*pyC

parser = argparse.ArgumentParser(description='Produce game time plot for advantage')
parser.add_argument('-p', '--pbp', default=None, required=False, type=str, help='PBP filename - needs csv extension')

params = parser.parse_args()

pbp_path = params.pbp

os.chdir('./gamestats/csvs/')
pbp = pd.read_csv(f'{pbp_path}')
os.chdir('../../')

cols = {'ANA':'#b85e0b','ARI':'#7d1db3','BOS':'#ffec00','BUF':'#b653fb','CAR':'#963e3e','CBJ':'#475483','CGY':'#b45300','CHI':'#8d6b0a','COL':'#6b051f','DAL':'#007f16','DET':'#ff0000','EDM':'#352247','FLA':'#77d200','LAK':'#380078','MIN':'#003a07','MTL':'#ec0365','NJD':'#ab0027','NSH':'#f3bf00','NYI':'#0078ff','NYR':'#07b182','OTT':'#805700','PHI':'#ff7c00','PIT':'#19bcd1','SEA':'#00c9b5','SJS':'#016072','STL':'#000df0','TBL':'#150078','TOR':'#363caf','VAN':'#5c6c98','VGK':'#bca900','WPG':'#140e6b','WSH':'#990276'}
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'font.family': 'sans-serif'})


### FUNCTIONS ###

def fo_parser(desc):

    winner, players = desc.split(' - ')
    zone = winner.split(' ')[2][:3]
    winner = winner[:3]
    player1, player2 = players.split(' vs ')
    team1 = player1[:3]
    team2 = player2[:3]

    if team1 == winner:
        winning_player = player1.split(' ')[-1]
        winning_team = team1
        losing_player = player2.split(' ')[-1]
        losing_team = team2
    elif team2 == winner:
        winning_player = player2.split(' ')[-1]
        winning_team = team2
        losing_player = player1.split(' ')[-1]
        losing_team = team1

    return zone, winning_player, winning_team, losing_player, losing_team

def advantager(bf, xf, fw, pd, g, t):

    return 0.1703894985932993*bf + 0.3879001066316915*xf + 0.12503971666619593*fw + 0.08937252123832294*pd + 0.1141218637729828*g + 0.11317629309750762*t

def percentager(a, b):
    if a + b == 0:
        c = 0.5
    else:
        c = a / (a + b)
    return c

def moving_average(x, w): # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    return np.convolve(x, np.ones(w), 'valid') / w

def plusminuser_weighted(on_ice, plus_team, minus_team, plus_weights, minus_weights, bf, xf, fw, pd, g, t):

    for i in on_ice:
        if i in plus_team.keys():
            plus_team[i] += 1 * advantager(bf, xf, fw, pd, g, t)
            plus_weights += 1
        elif i in minus_team.keys():
            minus_team[i] -= 1 * advantager(bf, xf, fw, pd, g, t)
            minus_weights += 1

    return(plus_team, minus_team, plus_weights, minus_weights)
#
# def combo_pm(on_ice, players, cf, sf, fw, pd, g, t, s):
#
#     if all(p in on_ice for p in players):
#         return advantager(cf, sf, fw, pd, g, t, s)
#     else:
#         return 0
#
# def combo_score(player_list, player_team):
#
#     events = pbp['Event']
#     descriptions = pbp['Description']
#
#     if player_team == home_team:
#         modifier = 1
#     elif player_team == away_team:
#         modifier = -1
#
#     combo = 0
#     n_events = 0
#
#     for j, e in enumerate(events):
#
#         on_ice = [pbp['awayPlayer1'][j], pbp['awayPlayer2'][j], pbp['awayPlayer3'][j], pbp['awayPlayer4'][j], pbp['awayPlayer5'][j], pbp['awayPlayer6'][j], pbp['homePlayer1'][j], pbp['homePlayer2'][j], pbp['homePlayer3'][j], pbp['homePlayer4'][j], pbp['homePlayer5'][j], pbp['homePlayer6'][j]]
#
#         if e == 'SHOT':
#             if descriptions[j].startswith(home_team):
#                 combo += combo_pm(on_ice, player_list, 0, 1, 0, 0, 0, 0, 0) * modifier
#                 n_events += 1
#             elif descriptions[j].startswith(away_team):
#                 combo -= combo_pm(on_ice, player_list, 0, 1, 0, 0, 0, 0, 0) * modifier
#                 n_events += 1
#
#         elif e == 'GOAL' and pers[j] != 5:
#             if descriptions[j].startswith(home_team):
#                 combo += combo_pm(on_ice, player_list, 0, 1, 0, 0, 0, 0, 0) * modifier
#                 n_events += 1
#             elif descriptions[j].startswith(away_team):
#                 combo -= combo_pm(on_ice, player_list, 0, 1, 0, 0, 0, 0, 0) * modifier
#                 n_events += 1
#
#         elif e == 'BLOCK' or e == 'MISS':
#             if descriptions[j].startswith(home_team):
#                 combo += combo_pm(on_ice, player_list, 1, 0, 0, 0, 0, 0, 0) * modifier
#                 n_events += 1
#             elif descriptions[j].startswith(away_team):
#                 combo -= combo_pm(on_ice, player_list, 1, 0, 0, 0, 0, 0, 0) * modifier
#                 n_events += 1
#
#         elif e == 'PENL':
#             p_team = descriptions[j].split(' ')[0]
#             mins = int(descriptions[j].split(' min)')[0].split('(')[-1])
#             if p_team == home_team:
#                 combo -= combo_pm(on_ice, player_list, 0, 0, 0, 1, 0, 0, 0) * modifier
#                 n_events += 1
#             elif p_team == away_team:
#                 combo += combo_pm(on_ice, player_list, 0, 0, 0, 1, 0, 0, 0) * modifier
#                 n_events += 1
#
#         elif e == 'FAC':
#             z, w, wt, l, lt = fo_parser(descriptions[j])
#             if wt == home_team:
#                 combo += combo_pm(on_ice, player_list, 0, 0, 1, 0, 0, 0, 0) * modifier
#                 n_events += 1
#             elif wt == away_team:
#                 combo -= combo_pm(on_ice, player_list, 0, 0, 1, 0, 0, 0, 0) * modifier
#                 n_events += 1
#
#         elif e == 'GIVE':
#             if descriptions[j].startswith(home_team):
#                 combo -= combo_pm(on_ice, player_list, 0, 0, 0, 0, 1, 0, 0) * modifier
#                 n_events += 1
#             elif descriptions[j].startswith(away_team):
#                 combo += combo_pm(on_ice, player_list, 0, 0, 0, 0, 1, 0, 0) * modifier
#                 n_events += 1
#
#         elif e == 'TAKE':
#             if descriptions[j].startswith(home_team):
#                 combo += combo_pm(on_ice, player_list, 0, 0, 0, 0, 0, 1, 0) * modifier
#                 n_events += 1
#             elif descriptions[j].startswith(away_team):
#                 combo -= combo_pm(on_ice, player_list, 0, 0, 0, 0, 0, 1, 0) * modifier
#                 n_events += 1
#
#     return combo / n_events


### ARRAYS ###

home_team = pbp['Home_Team'][0]
away_team = pbp['Away_Team'][0]
date = pbp['Date'][0]
file_out = date + '_' + away_team + 'at' + home_team

events = pbp['Event']
descriptions = pbp['Description']
secs = pbp['Seconds_Elapsed']
since_last = [s-secs[l-1] if l>1 else 0 for s,l in zip(secs,range(len(secs)))]
pers = pbp['Period']
strength = ['crap' if s!=s else s for s in pbp['Strength']]
zone = ['crap' if z!=z else z for z in pbp['Ev_Zone']]
shot_type = ['crap' if t!=t else t for t in pbp['Type']]
indices = range(len(pbp['Event']))
prev_event = ['crap' if pbp['Event'].iloc[i-1]!=pbp['Event'].iloc[i-1] else pbp['Event'].iloc[i-1] for e, i in zip(pbp['Event'], indices)]
xcoord = [0 if np.isnan(x)==True else x for x in pbp['xC']]
ycoord = [0 if np.isnan(y)==True else y for y in pbp['yC']]
prev_x = np.asarray([0 if np.isnan(pbp['xC'].iloc[i-1])==True else pbp['xC'].iloc[i-1] for e, i in zip(pbp['Event'], indices)])
prev_y = np.asarray([0 if np.isnan(pbp['yC'].iloc[i-1])==True else pbp['yC'].iloc[i-1] for e, i in zip(pbp['Event'], indices)])

for i, s in enumerate(strength):
    if s.startswith('-'):
        strength[i] = '-1x0'
for i, z in enumerate(zone):
    if type(z) != str:
        zone[i] = 'pass'
types_allowed = ['TIP-IN', 'DEFLECTED', 'WRIST SHOT', 'SLAP SHOT', 'SNAP SHOT', 'BACKHAND', 'WRAP-AROUND']
for i, s in enumerate(shot_type):
    if type(s) not in types_allowed:
        shot_type[i] = 'pass'
events_allowed = ['MISS', 'GIVE', 'CHL', 'SHOT', 'GOAL', 'STOP', 'GEND', 'FAC', 'BLOCK', 'HIT', 'PENL', 'TAKE', 'DELPEN', 'PSTR', 'PEND', 'EISTR']
for i, p in enumerate(prev_event):
    if type(p) not in events_allowed:
        prev_event[i] = 'pass'

strength_dict = {'0x5': 0, '5x6': 1, '5x2': 2, '4x4': 3, '5x3': 4, '5x1': 5, '4x3': 6, '4x6': 7, '6x6': 8, '5x0': 9, '4x5': 10, '6x1': 11, '5x7': 12, '5x4': 13, '1x5': 14, '2x5': 15, '3x5': 16, '2x3': 17, '0x0': 18, '1x1': 19, '1x0': 20, '8x5': 21, '-1x0': 22, '0x1': 23, '-1x-1': 24, '6x5': 25, '4x2': 26, '2x4': 27, '3x4': 28, '3x3': 29, '5x5': 30, 'crap':100, '6x4':101}
zone_dict = {'Off': 0, 'Neu': 1, 'Def': 2, 'crap':100}
shot_type_dict = {'TIP-IN': 1, 'DEFLECTED': 2, 'WRIST SHOT': 3, 'SLAP SHOT': 4, 'SNAP SHOT': 5, 'BACKHAND': 6, 'WRAP-AROUND': 7, 'crap':100, 'pass':101}
prev_event_dict = {'MISS': 0, 'GIVE': 1, 'CHL': 2, 'SHOT': 3, 'GOAL': 4, 'STOP': 5, 'GEND': 6, 'FAC': 7, 'BLOCK': 8, 'HIT': 9, 'PENL': 10, 'TAKE': 11, 'DELPEN': 12, 'PSTR': 13, 'PEND': 14, 'EISTR': 15, 'crap':100, 'pass':101}

# setting up categorical-->numerical variables
strength_nums = [strength_dict[x] for x in strength]
zone_nums = [zone_dict[x] for x in zone]
shot_type_nums = [shot_type_dict[x] for x in shot_type]
prev_event_nums = [prev_event_dict[x] for x in prev_event]

home_advantage = np.zeros(len(events))
away_advantage = np.zeros(len(events))
home_advantage_err = np.zeros(len(events))
away_advantage_err = np.zeros(len(events))
blocked_for_home = 0
blocked_for_away = 0
xG_for_home = 0
xG_for_away = 0
goals_for_home = 0
goals_for_away = 0
f_home = 0
f_away = 0
pmins_drawn_home = 0
pmins_drawn_away = 0
gives_against_home = 0
gives_against_away = 0
takes_for_home = 0
takes_for_away = 0

home_goals = []
away_goals = []

home_unfiltered = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(pbp['homePlayer1'], pbp['homePlayer2'], pbp['homePlayer3'], pbp['homePlayer4'], pbp['homePlayer5'], pbp['homePlayer6'])]
home_temp = np.unique(home_unfiltered)[:-1] # gets rid of nan values
score_placeholders = np.zeros(len(home_temp))
home_weighted = dict(zip(home_temp, score_placeholders))
home_weights = np.zeros(len(home_temp))

away_unfiltered = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(pbp['awayPlayer1'], pbp['awayPlayer2'], pbp['awayPlayer3'], pbp['awayPlayer4'], pbp['awayPlayer5'], pbp['awayPlayer6'])]
away_temp = np.unique(away_unfiltered)[:-1]
score_placeholders = np.zeros(len(away_temp))
away_weighted = dict(zip(away_temp, score_placeholders))
away_weights = np.zeros(len(away_temp))

n = 0 # for SEM


### SCRAPING ###

for j, e in enumerate(events):

    n += 1

    on_ice = [pbp['awayPlayer1'][j], pbp['awayPlayer2'][j], pbp['awayPlayer3'][j], pbp['awayPlayer4'][j], pbp['awayPlayer5'][j], pbp['awayPlayer6'][j], pbp['homePlayer1'][j], pbp['homePlayer2'][j], pbp['homePlayer3'][j], pbp['homePlayer4'][j], pbp['homePlayer5'][j], pbp['homePlayer6'][j]]

    if e == 'SHOT' or e == 'MISS':
        xg = xG(strength_nums[j], xcoord[j], ycoord[j], pers[j], secs[j], since_last[j], zone_nums[j], shot_type_nums[j], prev_event_nums[j], prev_x[j], prev_y[j])
        if descriptions[j].startswith(home_team):
            xG_for_home += 1
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 0, 1+xg, 0, 0, 0, 0)
        elif descriptions[j].startswith(away_team):
            xG_for_away += 1
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 0, 1+xg, 0, 0, 0, 0)

    elif e == 'GOAL' and pers[j] != 5:
        xg = xG(strength_nums[j], xcoord[j], ycoord[j], pers[j], secs[j], since_last[j], zone_nums[j], shot_type_nums[j], prev_event_nums[j], prev_x[j], prev_y[j])
        if descriptions[j].startswith(home_team):
            goals_for_home += 1
            home_goals.append((secs[j] + 1200*(pers[j]-1)) / 60)
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 0, 1+xg, 0, 0, 0, 0)
        elif descriptions[j].startswith(away_team):
            goals_for_away += 1
            away_goals.append((secs[j] + 1200*(pers[j]-1)) / 60)
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 0, 1+xg, 0, 0, 0, 0)

    elif e == 'BLOCK':
        if descriptions[j].startswith(home_team):
            blocked_for_home += 1
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 1, 0, 0, 0, 0, 0)
        elif descriptions[j].startswith(away_team):
            blocked_for_away += 1
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 1, 0, 0, 0, 0, 0)

    elif e == 'PENL':
        p_team = descriptions[j].split(' ')[0]
        mins = int(descriptions[j].split(' min)')[0].split('(')[-1])
        if p_team == home_team:
            pmins_drawn_away += mins
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 0, 0, 0, 1, 0, 0)
        elif p_team == away_team:
            pmins_drawn_home += mins
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 0, 0, 0, 1, 0, 0)

    elif e == 'FAC':
        z, w, wt, l, lt = fo_parser(descriptions[j])
        if wt == home_team:
            f_home += 1
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 0, 0, 1, 0, 0, 0)
        elif wt == away_team:
            f_away += 1
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 0, 0, 1, 0, 0, 0)

    elif e == 'GIVE':
        if descriptions[j].startswith(home_team):
            gives_against_away += 1
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 0, 0, 0, 0, 1, 0)
        elif descriptions[j].startswith(away_team):
            gives_against_home += 1
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 0, 0, 0, 0, 1, 0)

    elif e == 'TAKE':
        if descriptions[j].startswith(home_team):
            takes_for_home += 1
            plusminuser_weighted(on_ice, home_weighted, away_weighted, home_weights, away_weights, 0, 0, 0, 0, 0, 1)
        elif descriptions[j].startswith(away_team):
            takes_for_away += 1
            plusminuser_weighted(on_ice, away_weighted, home_weighted, away_weights, home_weights, 0, 0, 0, 0, 0, 1)

    bf_home = percentager(blocked_for_home, blocked_for_away)
    bf_away = percentager(blocked_for_away, blocked_for_home)
    xf_home = percentager(xG_for_home, xG_for_away)
    xf_away = percentager(xG_for_away, xG_for_home)
    gf_home = percentager(goals_for_home, goals_for_away)
    gf_away = percentager(goals_for_away, goals_for_home)
    fp_home = percentager(f_home, f_away)
    fp_away = percentager(f_away, f_home)
    pd_home = percentager(pmins_drawn_home, pmins_drawn_away)
    pd_away = percentager(pmins_drawn_away, pmins_drawn_home)
    ga_home = percentager(gives_against_home, gives_against_away)
    ga_away = percentager(gives_against_away, gives_against_home)
    tf_home = percentager(takes_for_home, takes_for_away)
    tf_away = percentager(takes_for_away, takes_for_home)
    # saves_home = sog_for_away - goals_for_away
    # saves_away = sog_for_home - goals_for_home
    # sp_home = percentager(saves_home, saves_away)
    # sp_away = percentager(saves_away, saves_home)

    home_advantage[j] = advantager(bf_home, xf_home, fp_home, pd_home, ga_home, tf_home)
    away_advantage[j] = advantager(bf_away, xf_away, fp_away, pd_away, ga_away, tf_away)

    home_advantage_err[j] = np.std((bf_home, xf_home, fp_home, pd_home, ga_home, tf_home)) / np.sqrt(n)
    away_advantage_err[j] = np.std((bf_away, xf_away, fp_away, pd_away, ga_away, tf_away)) / np.sqrt(n)

if 5 in list(set(pers)):
    so_goals = [d[:3] for (e, d, p) in zip(events, descriptions, pers) if p == 5 and e == 'GOAL']
    home_so = [t for t in so_goals if t == home_team]
    away_so = [t for t in so_goals if t == away_team]
    if len(home_so) > len(away_so):
        goals_for_home += 1
    else:
        goals_for_away += 1

home_wintype = ''
away_wintype = ''
max_time = 60
if 5 in list(set(pers)):
    max_time = 65
    if goals_for_home > goals_for_away:
        home_wintype = ' (SO)'
    else:
        away_wintype = ' (SO)'
elif 4 in list(set(pers)):
    max_time = 65
    if goals_for_home > goals_for_away:
        home_wintype = ' (OT)'
    else:
        away_wintype = ' (OT)'

minutes = (secs + 1200*(pers-1)) / 60

w = 10
ha = moving_average(home_advantage, w)
hae = moving_average(home_advantage_err, w)
aa = moving_average(away_advantage, w)
aae = moving_average(away_advantage_err, w)
events = moving_average(minutes, w)

home_goalies = [goalie for goalie in list(set(pbp['Home_Goalie'])) if str(goalie) != 'nan']
away_goalies = [goalie for goalie in list(set(pbp['Away_Goalie'])) if str(goalie) != 'nan']

i = 0
for k, v in home_weighted.items():
    if home_weights[i] != 0:
        home_weighted[k] = v / home_weights[i]
    i += 1
i = 0
for k, v in away_weighted.items():
    if away_weights[i] != 0:
        away_weighted[k] = v / away_weights[i]
    i += 1

for g in home_goalies:
    del home_weighted[g]
home_sorted = {k: v for k, v in sorted(home_weighted.items(), key=lambda item: item[1], reverse=True)}
for g in away_goalies:
    del away_weighted[g]
away_sorted = {k: v for k, v in sorted(away_weighted.items(), key=lambda item: item[1], reverse=True)}

# home_combos = np.zeros((len(home_sorted), len(home_sorted)))
# # home_swapsies = np.zeros((len(home_sorted), len(home_sorted)))
# for i, k1 in enumerate(home_sorted.keys()):
#     for j, k2 in enumerate(home_sorted.keys()):
#         if k1 != k2:
#             home_combos[i, j] = combo_score([k1,k2], home_team)
#         else:
#             home_combos[i, j] = home_sorted[k1]
#     # for j, k3 in enumerate(away_sorted.keys()):
#     #     home_swapsies[i, j] = combo_score([k1,k3], home_team)
#
# away_combos = np.zeros((len(away_sorted), len(away_sorted)))
# away_swapsies = np.zeros((len(away_sorted), len(away_sorted)))
# for i, k1 in enumerate(away_sorted.keys()):
#     for j, k2 in enumerate(away_sorted.keys()):
#         if k1 != k2:
#             away_combos[i, j] = combo_score([k1,k2], away_team)
#         else:
#             away_combos[i, j] = away_sorted[k1]
#     for j, k3 in enumerate(home_sorted.keys()):
#         away_swapsies[i, j] = combo_score([k1,k3], away_team)
#
# home_cols = [(0,0,0,1), (1,1,1,1), cl.to_rgba(cols[home_team])]
# home_map = cl.LinearSegmentedColormap.from_list('home map', home_cols, 300)
# away_cols = [(0,0,0,1), (1,1,1,1), cl.to_rgba(cols[away_team])]
# away_map = cl.LinearSegmentedColormap.from_list('away map', away_cols, 300)
# swap_cols = [cl.to_rgba(cols[home_team]), (1,1,1,1), cl.to_rgba('#5f5f5f')]
# swap_map = cl.LinearSegmentedColormap.from_list('swap map', swap_cols, 300)


#### https://stackoverflow.com/questions/35607818/index-a-2d-numpy-array-with-2-lists-of-indices
# clustest = spc.hierarchy.linkage(away_swapsies, method='complete', optimal_ordering=True)
# dn = spc.hierarchy.dendrogram(clustest, no_plot=True)
# sortby = dn['leaves']

# clustest2 = spc.hierarchy.linkage(away_swapsies.T, method='complete')
# dn2 = spc.hierarchy.dendrogram(clustest2, no_plot=True)
# sortby2 = dn2['leaves']

# cmnorm = cl.TwoSlopeNorm(vcenter=0)
# plt.imshow(away_swapsies.T[sortby2], cmap='twilight_shifted', norm=cmnorm)
# plt.show()

# sys.exit()

### FIGURES ###

# fig = plt.figure(figsize=(10,4), constrained_layout=True)

# plt.axhline(0.5, c='k', ls=':', alpha=0.5)
# plt.axvline(20, c='k', ls=':', alpha=0.5)
# plt.axvline(40, c='k', ls=':', alpha=0.5)
# plt.fill_between(minutes, home_advantage-home_advantage_err, home_advantage+home_advantage_err, color=cols[home_team], alpha=0.35, edgecolor=None)
# plt.fill_between(minutes, away_advantage-away_advantage_err, away_advantage+away_advantage_err, color=cols[away_team], alpha=0.35, edgecolor=None)
# plt.plot(minutes, home_advantage, '-', label=f'Home: {home_team} ({goals_for_home})', c=cols[home_team])
# plt.plot(minutes, away_advantage, '--', label=f'Away: {away_team} ({goals_for_away})', c=cols[away_team])
# for g in home_goals:
#     plt.axvline(g, c=cols[f'{home_team}'], ls='-')
# for g in away_goals:
#     plt.axvline(g, c=cols[f'{away_team}'], ls='--')
# plt.legend(loc='upper right')
# plt.xlabel('minutes elapsed')
# plt.ylabel('statistical advantage')
# plt.xlim(0, 60)

# plt.savefig(f'{pbp_path}_raw.png')
# plt.close()

os.chdir('./hockeyfigs')

fig = plt.figure(figsize=(10,4), constrained_layout=True)

plt.axhline(0.5, c='k', ls=':', alpha=0.5)
plt.axvline(20, c='k', ls=':', alpha=0.5)
plt.axvline(40, c='k', ls=':', alpha=0.5)
plt.fill_between(events, ha-hae, ha+hae, color=cols[home_team], alpha=0.35, edgecolor=None)
plt.fill_between(events, aa-aae, aa+aae, color=cols[away_team], alpha=0.35, edgecolor=None)
plt.plot(events, ha, '-', label=f'{home_team} ({goals_for_home}){home_wintype}', c=cols[home_team])
plt.plot(events, aa, '--', label=f'{away_team} ({goals_for_away}){away_wintype}', c=cols[away_team])
for g in home_goals:
    plt.axvline(g, c=cols[home_team], ls='-', lw=1.5)
for g in away_goals:
    plt.axvline(g, c=cols[away_team], ls='--', lw=1.5)
plt.legend(loc='upper right')
plt.xlabel('minutes elapsed')
plt.ylabel('statistical advantage')
plt.xlim(0, max_time)
plt.title(f'{away_team} @ {home_team} // {date}')

plt.savefig(f'{file_out}_tide.png')
plt.close()

# fig, ax = plt.subplots(1, constrained_layout=True, figsize=(8,8))
#
# cmnorm = cl.TwoSlopeNorm(vcenter=0)
# home_labels = [capwords(k) for k in home_sorted.keys()]
#
# vibes = ax.imshow(home_combos, cmap=home_map, norm=cmnorm)
#
# home_ticks = [np.min(home_combos)] + list(range(int(np.trunc(np.min(home_combos))), int(np.trunc(np.max(home_combos)))+1)) + [np.max(home_combos)]
# home_ticklabels = [away_team] + list(range(int(np.trunc(np.min(home_combos))), int(np.trunc(np.max(home_combos)))+1)) + [home_team]
# cb = plt.colorbar(vibes, label='player score', ticks=home_ticks, fraction=0.05, pad=0.04)
# cb.ax.set_yticklabels(home_ticklabels)
#
# ax.set_xticks(range(len(home_labels)))
# ax.set_xticklabels(home_labels, rotation=90)
# ax.set_yticks(range(len(home_labels)))
# ax.set_yticklabels(home_labels)
# ax.set_title(fr'{away_team} @ $\textbf{{{home_team}}}$ // {date}')
#
# plt.savefig(f'{file_out}_home.png')
# plt.close()
#
# fig, ax = plt.subplots(1, constrained_layout=True, figsize=(8,8))
#
# cmnorm = cl.TwoSlopeNorm(vcenter=0)
# away_labels = [capwords(k) for k in away_sorted.keys()]
#
# vibes = ax.imshow(away_combos, cmap=away_map, norm=cmnorm)
#
# away_ticks = [np.min(away_combos)] + list(range(int(np.trunc(np.min(away_combos))), int(np.trunc(np.max(away_combos)))+1)) + [np.max(away_combos)]
# away_ticklabels = [home_team] + list(range(int(np.trunc(np.min(away_combos))), int(np.trunc(np.max(away_combos)))+1)) + [away_team]
# cb = plt.colorbar(vibes, label='player score', ticks=away_ticks, fraction=0.05, pad=0.04)
# cb.ax.set_yticklabels(away_ticklabels)
#
# ax.set_xticks(range(len(away_labels)))
# ax.set_xticklabels(away_labels, rotation=90)
# ax.set_yticks(range(len(away_labels)))
# ax.set_yticklabels(away_labels)
# ax.set_title(fr'$\textbf{{{away_team}}}$ @ {home_team} // {date}')
#
# plt.savefig(f'{file_out}_away.png')
# plt.close()
#
# fig, ax = plt.subplots(1, constrained_layout=True, figsize=(8,8))
#
# cmnorm = cl.TwoSlopeNorm(vcenter=0)
#
# vibes = ax.imshow(away_swapsies, cmap='twilight_shifted', norm=cmnorm)
#
# swap_ticks = [np.min(away_swapsies)] + list(range(int(np.trunc(np.min(away_swapsies))), int(np.trunc(np.max(away_swapsies)))+1)) + [np.max(away_swapsies)]
# swap_ticklabels = [home_team] + list(range(int(np.trunc(np.min(away_swapsies))), int(np.trunc(np.max(away_swapsies)))+1)) + [away_team]
# cb = plt.colorbar(vibes, label='player score', ticks=swap_ticks, fraction=0.05, pad=0.04)
# cb.ax.set_yticklabels(swap_ticklabels)
#
# ax.set_xticks(range(len(home_labels)))
# ax.set_xticklabels(home_labels, rotation=90)
# ax.set_yticks(range(len(away_labels)))
# ax.set_yticklabels(away_labels)
# ax.set_title(f'{away_team} @ {home_team} // {date}')
#
# plt.savefig(f'{file_out}_match.png')
# plt.close()

os.chdir('..')
