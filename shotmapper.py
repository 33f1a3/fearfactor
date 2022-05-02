import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.text import TextPath
import matplotlib as mpl
from string import *
import argparse, os, sys, joblib

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
mpl.rcParams.update({'font.size': 12})
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
shot_type = [t for e, t in zip(pbp['Event'], pbp['Type']) if e == 'GOAL' or e == 'SHOT' or e == 'MISS']
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

angles = np.degrees(np.arctan((xc_plot)/yc_plot))
angles[np.isnan(angles)] = 0

marks = [(3,0,a+90) if y>=0 and t==away_team #and p<5
    else (3,0,a-90) if y>=0 #and p<5
    else (3,0,a) #if p<5
    # else (3,0,a+90) if y>=0 and t==away_team
    # else (3,0,a-90) if y>=0
    # else (3,0,a)
    for a, t, y, p in zip(angles, event_team, yc_plot, per)]
marks = np.asarray(marks)

fig, ax = plt.subplots(1, figsize=(8,4), constrained_layout=True)

for i in range(len(xc_plot[events=='SHOT'])):

    strstr1 = int(strength[events=='SHOT'][i].split('x')[0])
    strstr2 = int(strength[events=='SHOT'][i].split('x')[1])

    if strstr1 > strstr2:
        if event_team[events=='SHOT'][i]==home_team:
            ax.scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               edgecolor=cols[event_team[events=='SHOT'][i]], facecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i])
        elif event_team[events=='SHOT'][i]==away_team:
            ax.scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               facecolor=cols[event_team[events=='SHOT'][i]], edgecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i])
    elif strstr1 < strstr2:
        if event_team[events=='SHOT'][i]==home_team:
            ax.scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               facecolor=cols[event_team[events=='SHOT'][i]], edgecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i])
        elif event_team[events=='SHOT'][i]==away_team:
            ax.scatter(xc_plot[events=='SHOT'][i],
               yc_plot[events=='SHOT'][i],
               s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=2,
               edgecolor=cols[event_team[events=='SHOT'][i]], facecolor='none',
               alpha=1, marker=marks[events=='SHOT'][i])
    elif per[events=='SHOT'][i] == 5:
        ax.scatter(xc_plot[events=='SHOT'][i],
           yc_plot[events=='SHOT'][i],
           s=(all_xg[events=='SHOT'][i]+0.01)*3000, linewidth=1,
           edgecolor='k', facecolor='none',
           alpha=1, marker=marks[events=='SHOT'][i])
    else:
         ax.scatter(xc_plot[events=='SHOT'][i],
             yc_plot[events=='SHOT'][i],
             s=(all_xg[events=='SHOT'][i]+0.01)*3000,
             facecolor=cols[event_team[events=='SHOT'][i]], edgecolor='none',
             alpha=0.5, marker=marks[events=='SHOT'][i])

# if per[events=='SHOT'][i] != 5:

for i in range(len(xc_plot[events=='MISS'])):

    strstr1 = int(strength[events=='MISS'][i].split('x')[0])
    strstr2 = int(strength[events=='MISS'][i].split('x')[1])

    if strstr1 > strstr2:
        if event_team[events=='MISS'][i]==home_team:
            ax.scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               edgecolor='#525252', facecolor='none',
               alpha=0.6, marker=marks[events=='MISS'][i])
        elif event_team[events=='MISS'][i]==away_team:
            ax.scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               facecolor='#525252', edgecolor='none',
               alpha=1, marker=marks[events=='MISS'][i])
    elif strstr1 < strstr2:
        if event_team[events=='MISS'][i]==home_team:
            ax.scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               facecolor='#525252', edgecolor='none',
               alpha=1, marker=marks[events=='MISS'][i])
        elif event_team[events=='MISS'][i]==away_team:
            ax.scatter(xc_plot[events=='MISS'][i],
               yc_plot[events=='MISS'][i],
               s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=2,
               edgecolor='#525252', facecolor='none',
               alpha=0.6, marker=marks[events=='MISS'][i])
    elif per[events=='MISS'][i] == 5:
        ax.scatter(xc_plot[events=='MISS'][i],
                   yc_plot[events=='MISS'][i],
                   s=(all_xg[events=='MISS'][i]+0.01)*3000, linewidth=1,
                   edgecolor='k', facecolor='none', linestyle='--',
                   alpha=1, marker=marks[events=='MISS'][i])
    else:
        ax.scatter(xc_plot[events=='MISS'][i],
            yc_plot[events=='MISS'][i],
            s=(all_xg[events=='MISS'][i]+0.01)*3000,
            facecolor='#525252', edgecolor='none',
            alpha=0.4, marker=marks[events=='MISS'][i])

for g in range(len(xc_plot[(events=='GOAL')])):

    so_increment = 1

    scorer = descriptions[events=='GOAL'][g].split(' ')[2]
    if '(' in scorer:
        scorer = scorer.split('(')[0].capitalize()
    else:
        scorer = scorer.split(',')[0].capitalize()

    if per[events=='GOAL'][g] == 4:
        mark = 'OT'
        msize = 6000
    elif per[events=='GOAL'][g] == 5:
        mark = 'SO' + f'{so_increment}'
        so_increment += 1
        msize = 5000
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

    ax.scatter(xc_plot[events=='GOAL'][g],
           yc_plot[events=='GOAL'][g],
#            s=(all_xg[events=='GOAL'][g]+0.01)*1750,
           # c=cols[event_team[events=='GOAL'][g]],
           facecolor='k',
           edgecolor='none', #linewidth=0.6,
           marker=textmark, s=msize)

    xgh = all_xg[events=='GOAL'][g]
    ax.annotate(f'{str(mark)} - {scorer}{stren} ({xgh:.2f} xG)', (-25, (g+1)*-5 + 40))

ax.set(xlim=(-100,100), ylim=(-50,50), xticks=[], yticks=[],
    title=f'{away_team} @ {home_team} // {date}')

plt.savefig(f'{date}_{away_team}@{home_team}.png')
