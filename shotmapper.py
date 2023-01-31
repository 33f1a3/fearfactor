import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from hockey_rink import NHLRink
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.text import TextPath
import matplotlib as mpl
from string import *
import argparse, os, sys, joblib, warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

forest = joblib.load('fearfactorxg.joblib')
importances = forest.feature_importances_

def xG(s, xC, yC, p, sinp, ssle, z, st, pe, pxC, pyC):
    return importances[0]*s + importances[1]*xC + importances[2]*yC + importances[3]*p + importances[4]*sinp + importances[5]*ssle + importances[6]*z + importances[7]*st + importances[8]*pe + importances[9]*pxC + importances[10]*pyC

### I/O ###

parser = argparse.ArgumentParser(description='Produce xG shot map')
parser.add_argument('-p', '--pbp', default=None, required=False, type=str, help='PBP filename - needs csv extension')

params = parser.parse_args()

pbp_path = params.pbp

os.chdir('./gamestats/csvs/')
pbp = pd.read_csv(f'{pbp_path}')
os.chdir('../../')

cols = {'ANA':'#b85e0b','ARI':'#7d1db3','BOS':'#ffec00','BUF':'#b653fb','CAR':'#963e3e','CBJ':'#475483','CGY':'#b45300','CHI':'#8d6b0a','COL':'#6b051f','DAL':'#007f16','DET':'#ff0000','EDM':'#352247','FLA':'#77d200','LAK':'#380078','MIN':'#003a07','MTL':'#ec0365','NJD':'#ab0027','NSH':'#f3bf00','NYI':'#0078ff','NYR':'#07b182','OTT':'#805700','PHI':'#ff7c00','PIT':'#19bcd1','SEA':'#00c9b5','SJS':'#016072','STL':'#000df0','TBL':'#150078','TOR':'#363caf','VAN':'#5c6c98','VGK':'#bca900','WPG':'#140e6b','WSH':'#990276'}
# mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'font.family': 'Inconsolata'})

home_team = pbp['Home_Team'][0]
away_team = pbp['Away_Team'][0]
date = pbp['Date'][0]

events = [e for e in pbp['Event'] if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
event_team = [t for e, t in zip(pbp['Event'], pbp['Ev_Team']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
descriptions = [d for e, d in zip(pbp['Event'], pbp['Description']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
strength = [s for e, s in zip(pbp['Event'], pbp['Strength']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
xcoord = np.asarray([x for e, x in zip(pbp['Event'], pbp['xC']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS'])
ycoord = np.asarray([y for e, y in zip(pbp['Event'], pbp['yC']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS'])
per_time = [t for e, t in zip(pbp['Event'], pbp['Time_Elapsed']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
per = [p for e, p in zip(pbp['Event'], pbp['Period']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
secs = [s for e, s in zip(pbp['Event'], pbp['Seconds_Elapsed']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
zone = [z for e, z in zip(pbp['Event'], pbp['Ev_Zone']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
shot_type = ['TIP-IN' if 'Poke' in d else t for d, e, t in zip(pbp['Description'], pbp['Event'], pbp['Type']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
indices = range(len(pbp['Event']))
prev_event = [pbp['Event'].iloc[i-1] for e, i in zip(pbp['Event'], indices) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
prev_x = np.asarray([pbp['xC'].iloc[i-1] for e, i in zip(pbp['Event'], indices) if e == 'GOAL' or e == 'SHOT' or e == 'MISS'])
prev_y = np.asarray([pbp['yC'].iloc[i-1] for e, i in zip(pbp['Event'], indices) if e == 'GOAL' or e == 'SHOT' or e == 'MISS'])

xcoord[np.isnan(xcoord)] = 0
ycoord[np.isnan(ycoord)] = 0
prev_x[np.isnan(prev_x)] = 0
prev_y[np.isnan(prev_y)] = 0

since_last = [s-secs[l-1] if s>=secs[l-1] else -1 for s,l in zip(secs,range(len(secs)))]

strength_dict = {'0x5': 0, '5x6': 1, '5x2': 2, '4x4': 3, '5x3': 4, '5x1': 5, '4x3': 6, '4x6': 7, '6x6': 8, '5x0': 9, '4x5': 10, '6x1': 11, '5x7': 12, '5x4': 13, '1x5': 14, '2x5': 15, '3x5': 16, '2x3': 17, '0x0': 18, '1x1': 19, '1x0': 20, '8x5': 21, '-1x0': 22, '0x1': 23, '-1x-1': 24, '6x5': 25, '4x2': 26, '2x4': 27, '3x4': 28, '3x3': 29, '5x5': 30}
zone_dict = {'Off': 0, 'Neu': 1, 'Def': 2}
shot_type_dict = {'TIP-IN': 1, 'DEFLECTED': 2, 'WRIST SHOT': 3, 'SLAP SHOT': 4, 'SNAP SHOT': 5, 'BACKHAND': 6, 'WRAP-AROUND': 7}
prev_event_dict = {'MISS': 0, 'GIVE': 1, 'CHL': 2, 'SHOT': 3, 'GOAL': 4, 'STOP': 5, 'GEND': 6, 'FAC': 7, 'BLOCK': 8, 'HIT': 9, 'PENL': 10, 'TAKE': 11, 'DELPEN': 12, 'PSTR': 13, 'PEND': 14, 'EISTR': 15}

# setting up categorical-->numerical variables
strength_nums = [strength_dict[x] for x in strength]
zone_nums = [zone_dict[x] for x in zone]
shot_type_nums = [shot_type_dict[x] for x in shot_type]
prev_event_nums = [prev_event_dict[x] for x in prev_event]

all_xg = forest.predict(np.array((strength_nums, xcoord, ycoord, per, secs, since_last, zone_nums,
              shot_type_nums, prev_event_nums, prev_x, prev_y)).T)

# dealing with evilness
if home_team == 'N.J':
    home_team = 'NJD'
    event_team = ['NJD' if t == 'N.J' else t for t in event_team]

# setting up data for plotting
per = np.asarray(per)

xc_plot = [x if p%2==0 else -x for x, p in zip(xcoord,per)]
xc_plot = np.asarray(xc_plot)

yc_plot = [y if p%2==0 else -y for y, p in zip(ycoord,per)]
yc_plot = np.asarray(yc_plot)

event_team = np.asarray(event_team)
events = np.asarray(events)
all_xg = np.asarray(all_xg)
descriptions = np.asarray(descriptions)
strength = np.asarray(strength)

x_mod = np.ones(len(xc_plot)) + np.sign(xc_plot)*15
angles = np.degrees(np.arctan((xc_plot-x_mod)/yc_plot))
angles[np.isnan(angles)] = 0

marks = [(3,0,a+90) if y>=0 and t==away_team #and p<5
    else (3,0,a-90) if y>=0 #and p<5
    else (3,0,a) #if p<5
    # else (3,0,a+90) if y>=0 and t==away_team
    # else (3,0,a-90) if y>=0
    # else (3,0,a)
    for a, t, y, p in zip(angles, event_team, yc_plot, per)]
marks = np.asarray(marks)

### ACTUAL PLOTTING

mosaic = """
    AA
    xy
    BB
"""

# fig, ax = plt.subplots(1, figsize=(8,4), constrained_layout=True)
goalcount = len(event_team[events=='GOAL'])
heightmod = goalcount*0.2

ax = plt.figure(figsize=(9,7+heightmod), constrained_layout=True).subplot_mosaic(mosaic,
    gridspec_kw={'height_ratios':[1,0.3*heightmod,0.5], 'width_ratios':[1,0.75]})


rink = NHLRink(line_thickness=0.001, line_color='k',
               boards={"thickness":0.2, "alpha":0.3},
               blue_line={"color":'k', "alpha":0.3, "length":0.2},
               red_line={"color":'k', "alpha":0.3, "length":0.3},
               center_dot={"visible":False}, faceoff_dot={"visible":False},
               faceoff_circle={"alpha":0.25}, center_circle={"color":'k', "alpha":0.25},
               trapezoid={"visible":False}, faceoff_lines={"visible":False}, ref_circle={"visible":False},
               crease={"visible":False}, net={"visible":False}, crossbar={"visible":False},
               crease_outline={"visible":True, "alpha":0.4, "thickness":0.01},
               goal_line={"alpha":0.3, "length":0.2})
ax['A'] = rink.draw(ax=ax['A'])

for i in range(len(xc_plot[events=='SHOT'])):

    strstr1 = int(strength[events=='SHOT'][i].split('x')[0])
    strstr2 = int(strength[events=='SHOT'][i].split('x')[1])

    if strstr1 > strstr2:
        if event_team[events=='SHOT'][i]==home_team:
            ax['A'].scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               edgecolor=cols[event_team[events=='SHOT'][i]], facecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i], zorder=10000)
        elif event_team[events=='SHOT'][i]==away_team:
            ax['A'].scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               edgecolor=cols[event_team[events=='SHOT'][i]], facecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i], linestyle='dotted', zorder=10000)
    elif strstr1 < strstr2:
        if event_team[events=='SHOT'][i]==home_team:
            ax['A'].scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               edgecolor=cols[event_team[events=='SHOT'][i]], facecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i], linestyle='dotted', zorder=10000)
        elif event_team[events=='SHOT'][i]==away_team:
            ax['A'].scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               edgecolor=cols[event_team[events=='SHOT'][i]], facecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i], zorder=10000)
    elif per[events=='SHOT'][i] == 5:
        ax['A'].scatter(xc_plot[events=='SHOT'][i],
           yc_plot[events=='SHOT'][i],
           s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=1,
           edgecolor='k', facecolor='none',
           alpha=1, marker=marks[events=='SHOT'][i], zorder=10000)
    else:
         ax['A'].scatter(xc_plot[events=='SHOT'][i],
             yc_plot[events=='SHOT'][i],
             s=(all_xg[events=='SHOT'][i]+0.01)*3000,
             facecolor=cols[event_team[events=='SHOT'][i]], edgecolor='none',
             alpha=0.5, marker=marks[events=='SHOT'][i], zorder=10000)

for i in range(len(xc_plot[events=='MISS'])):

    strstr1 = int(strength[events=='MISS'][i].split('x')[0])
    strstr2 = int(strength[events=='MISS'][i].split('x')[1])

    if strstr1 > strstr2:
        if event_team[events=='MISS'][i]==home_team:
            ax['A'].scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               edgecolor='#525252', facecolor='none',
               alpha=0.6, marker=marks[events=='MISS'][i], zorder=10000)
        elif event_team[events=='MISS'][i]==away_team:
            ax['A'].scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               edgecolor='#525252', facecolor='none', linestyle='dotted',
               alpha=0.6, marker=marks[events=='MISS'][i], zorder=10000)
    elif strstr1 < strstr2:
        if event_team[events=='MISS'][i]==home_team:
            ax['A'].scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               edgecolor='#525252', facecolor='none', linestyle='dotted',
               alpha=0.6, marker=marks[events=='MISS'][i], zorder=10000)
        elif event_team[events=='MISS'][i]==away_team:
            ax['A'].scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               edgecolor='#525252', facecolor='none',
               alpha=0.6, marker=marks[events=='MISS'][i], zorder=10000)
    elif per[events=='MISS'][i] == 5:
        ax['A'].scatter(xc_plot[events=='MISS'][i],
                   yc_plot[events=='MISS'][i],
                   s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=1,
                   edgecolor='k', facecolor='none', linestyle='--',
                   alpha=1, marker=marks[events=='MISS'][i], zorder=10000)
    else:
        ax['A'].scatter(xc_plot[events=='MISS'][i],
            yc_plot[events=='MISS'][i],
            s=(all_xg[events=='MISS'][i]+0.01)*3000,
            facecolor='#525252', edgecolor='none',
            alpha=0.4, marker=marks[events=='MISS'][i], zorder=10000)

home_count = 0
away_count = 0
for g in range(len(xc_plot[(events=='GOAL')])):

    so_increment = 1

    if event_team[events=='GOAL'][g] == home_team:
        home_count += 1
    else:
        away_count += 1

    scorer = descriptions[events=='GOAL'][g].split(' ')[2]
    if '(' in scorer:
        scorer = scorer.split('(')[0].capitalize()
    else:
        scorer = scorer.split(',')[0].capitalize()

    if per[events=='GOAL'][g] == 4:
        mark = 'OT'
        msize = 4000
    elif per[events=='GOAL'][g] == 5:
        mark = 'SO' + f'{so_increment}'
        so_increment += 1
        msize = 4000
    else:
        mark = g+1
        msize = 1000

    strstr1 = int(strength[events=='GOAL'][g].split('x')[0])
    strstr2 = int(strength[events=='GOAL'][g].split('x')[1])
    if strstr1 > strstr2 and event_team[events=='GOAL'][g]==home_team:
        stren = ' PPG'
    elif strstr1 > strstr2 and event_team[events=='GOAL'][g]==away_team:
        stren = ' SHG'
    elif strstr1 < strstr2  and event_team[events=='GOAL'][g]==home_team:
        stren = ' SHG'
    elif strstr1 < strstr2 and event_team[events=='GOAL'][g]==away_team:
        stren = ' PPG'
    else:
        stren = ''

    textmark = TextPath((0,0), str(mark))

    ax['A'].scatter(xc_plot[events=='GOAL'][g],
           yc_plot[events=='GOAL'][g],
#            s=(all_xg[events=='GOAL'][g]+0.01)*1750,
           # c=cols[event_team[events=='GOAL'][g]],
           facecolor='k',
           edgecolor='none', #linewidth=0.6,
           marker=textmark, s=msize, zorder=10000)

    xgh = all_xg[events=='GOAL'][g]
    ax['y'].annotate(f'{str(mark)} - {scorer}{stren} - {xgh:.2f} xG', (0, goalcount-g-0.5))

winning_team = ''
if home_count > away_count:
    winning_team = home_team
else:
    winning_team = away_team

qualifier = ''
if max(per) == 4:
    qualifier = ' (OT)'
elif max(per) == 5:
    qualifier = ' (SO)'

ax['A'].set(xlim=(-100,107), ylim=(-42.5,42.5), xticks=[], yticks=[],
    title=f'{away_team} {away_count} @ {home_team} {home_count}{qualifier} // {date}')
ax['y'].set(ylim=(0,goalcount))
ax['y'].axis('off')

ax['x'].scatter(0.1, 0.8, marker=(3,0,-90),
    facecolor=cols[winning_team], s=500, alpha=0.5, edgecolor='none')
ax['x'].scatter(0.1, 0.5, marker=(3,0,-90),
    facecolor='#525252', s=500, alpha=0.5, edgecolor='none')
ax['x'].scatter(0.1, 0.2, marker=(3,0,-90), linewidth=1,
    facecolor='none', s=500, alpha=1, edgecolor='k')

ax['x'].scatter(0.6, 0.8, marker=(3,0,-90), linewidth=2,
    edgecolor=cols[winning_team], s=500, alpha=1, facecolor='none')
ax['x'].scatter(0.6, 0.5, marker=(3,0,-90), linewidth=2, linestyle='dotted',
    edgecolor=cols[winning_team], s=500, alpha=1, facecolor='none')

ax['x'].annotate('shot on goal', (0.18, 0.75))
ax['x'].annotate('missed shot', (0.18, 0.45))
ax['x'].annotate('shootout shot', (0.18, 0.15))
ax['x'].annotate('power play', (0.68, 0.75))
ax['x'].annotate('shorthanded', (0.68, 0.45))

ax['x'].set(xlim=(0,1), ylim=(0-0.3*heightmod,1))
ax['x'].axis('off')



### AND NOW FOR ADVANTAGE ###

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

strength_dict = {'0x5': 0, '5x6': 1, '5x2': 2, '4x4': 3, '5x3': 4, '5x1': 5, '4x3': 6, '4x6': 7, '6x6': 8, '5x0': 9, '4x5': 10, '6x1': 11, '5x7': 12, '5x4': 13, '1x5': 14, '2x5': 15, '3x5': 16, '2x3': 17, '0x0': 18, '1x1': 19, '1x0': 20, '8x5': 21, '-1x0': 22, '0x1': 23, '-1x-1': 24, '6x5': 25, '4x2': 26, '2x4': 27, '3x4': 28, '3x3': 29, '5x5': 30, 'crap':100, '6x4':101, '3x2':102}
zone_dict = {'Off': 0, 'Neu': 1, 'Def': 2, 'crap':100}
shot_type_dict = {'TIP-IN': 1, 'DEFLECTED': 2, 'WRIST SHOT': 3, 'SLAP SHOT': 4, 'SNAP SHOT': 5, 'BACKHAND': 6, 'WRAP-AROUND': 7, 'crap':100, 'pass':101}
prev_event_dict = {'MISS': 0, 'GIVE': 1, 'CHL': 2, 'SHOT': 3, 'GOAL': 4, 'STOP': 5, 'GEND': 6, 'FAC': 7, 'BLOCK': 8, 'HIT': 9, 'PENL': 10, 'TAKE': 11, 'DELPEN': 12, 'PSTR': 13, 'PEND': 14, 'EISTR': 15, 'crap':100, 'pass':101}

# setting up categorical-->numerical variables
strength_nums = [strength_dict[x] for x in strength]
zone_nums = [zone_dict[x] for x in zone]
shot_type_nums = [shot_type_dict[x] for x in shot_type]
prev_event_nums = [prev_event_dict[x] for x in prev_event]

# dealing with evilness
if home_team == 'N.J':
    home_team = 'NJD'

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

ax['B'].axhline(0.5, c='k', ls=':', alpha=0.5)
ax['B'].axvline(20, c='k', ls=':', alpha=0.5)
ax['B'].axvline(40, c='k', ls=':', alpha=0.5)
ax['B'].fill_between(events, ha-hae, ha+hae, color=cols[home_team], alpha=0.35, edgecolor=None)
ax['B'].fill_between(events, aa-aae, aa+aae, color=cols[away_team], alpha=0.35, edgecolor=None)
ax['B'].plot(events, ha, '-', label=f'{home_team}', c=cols[home_team])
ax['B'].plot(events, aa, '--', label=f'{away_team}', c=cols[away_team])
for g in home_goals:
    ax['B'].axvline(g, c=cols[home_team], ls='-', lw=1.5)
for g in away_goals:
    ax['B'].axvline(g, c=cols[away_team], ls='--', lw=1.5)
ax['B'].legend(loc='upper right')
# plt.xlabel('minutes elapsed')
# plt.ylabel('statistical advantage')
ax['B'].set(xlim=(0, max_time), xlabel='minutes elapsed', ylabel='advantage')
ax['B'].spines['top'].set_visible(False)
ax['B'].spines['right'].set_visible(False)
ax['B'].spines['bottom'].set_visible(False)
ax['B'].spines['left'].set_visible(False)
ax['B'].get_yaxis().set_ticks([])
ax['B'].yaxis.labelpad = 10

os.chdir('./hockeyfigs')
plt.savefig(f'{date}_{away_team}@{home_team}.png')
os.chdir('..')
