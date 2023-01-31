import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hockeysuite as hs
import argparse, joblib, sys

parser = argparse.ArgumentParser(description='Produce game time plot for advantage')
parser.add_argument('-p', '--pbp', required=True, type=str, help='PBP filename - no extention')
parser.add_argument('-c', '--combo', required=True, type=str, nargs='*', help='List of players - full names')
parser.add_argument('-t', '--team', required=True, type=str, help='3-letter team code for players')

params = parser.parse_args()

pbp_path = params.pbp
player_list = params.combo
player_team = params.team

pbp = pd.read_csv(f'{pbp_path}.csv')

forest = joblib.load('fearfactorxg.joblib')
importances = forest.feature_importances_

def xG(s, xC, yC, p, sinp, ssle, z, st, pe, pxC, pyC):
    return importances[0]*s + importances[1]*xC + importances[2]*yC + importances[3]*p + importances[4]*sinp + importances[5]*ssle + importances[6]*z + importances[7]*st + importances[8]*pe + importances[9]*pxC + importances[10]*pyC

cols = {'ANA':'#694632','ARI':'#7d1db3','BOS':'#ffec00','BUF':'#b653fb','CAR':'#963e3e','CBJ':'#475483','CGY':'#b45300','CHI':'#8d6b0a','COL':'#6b051f','DAL':'#007f16','DET':'#ff0000','EDM':'#55483d','FLA':'#77d200','L.A':'#1f1f1f','MIN':'#003a07','MTL':'#ec0365','N.J':'#ab0027','NSH':'#f3bf00','NYI':'#0078ff','NYR':'#07b182','OTT':'#646464','PHI':'#ff7c00','PIT':'#19bcd1','SEA':'#05f6db','SJS':'#016072','STL':'#000df0','T.B':'#150078','TOR':'#363caf','VAN':'#5c6c98','VGK':'#bca900','WPG':'#928d92','WSH':'#4d0069'}

def fo_parser(desc):
    
    try:
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

    except IndexError:
        zone = None
        winning_player = None
        winning_team = None
        losing_player = None
        losing_team = None
        
    return zone, winning_player, winning_team, losing_player, losing_team

def advantager(bf, xf, fw, pd, g, t):

    return 0.1703894985932993*bf + 0.3879001066316915*xf + 0.12503971666619593*fw + 0.08937252123832294*pd + 0.1141218637729828*g + 0.11317629309750762*t

def percentager(a, b):
    if a + b == 0:
        c = 0.5
    else:
        c = a / (a + b)
    return c

def plusminuser(on_ice, plus_team, minus_team):
    
    for i in on_ice:
        if i in plus_team.keys():
            plus_team[i] += 1
        elif i in minus_team.keys():
            minus_team[i] -= 1
            
    return(plus_team, minus_team)

def plusminuser_weighted(on_ice, plus_team, minus_team, plus_weights, minus_weights, bf, xf, fw, pd, g, t):

    for i in on_ice:
        if i in plus_team.keys():
            plus_team[i] += 1 * advantager(bf, xf, fw, pd, g, t)
            plus_weights += 1
        elif i in minus_team.keys():
            minus_team[i] -= 1 * advantager(bf, xf, fw, pd, g, t)
            minus_weights += 1

    return(plus_team, minus_team, plus_weights, minus_weights)

def combo_pm(on_ice, players, bf, xf, fw, pd, g, t):
    
    if all(p in on_ice for p in players):
        return advantager(bf, xf, fw, pd, g, t)
    else:
        return 0

#players
a1 = [p for p, h, a in zip(pbp['awayPlayer1'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
a2 = [p for p, h, a in zip(pbp['awayPlayer2'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
a3 = [p for p, h, a in zip(pbp['awayPlayer3'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
a4 = [p for p, h, a in zip(pbp['awayPlayer4'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
a5 = [p for p, h, a in zip(pbp['awayPlayer5'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
a6 = [p for p, h, a in zip(pbp['awayPlayer6'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
h1 = [p for p, h, a in zip(pbp['homePlayer1'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
h2 = [p for p, h, a in zip(pbp['homePlayer2'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
h3 = [p for p, h, a in zip(pbp['homePlayer3'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
h4 = [p for p, h, a in zip(pbp['homePlayer4'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
h5 = [p for p, h, a in zip(pbp['homePlayer5'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
h6 = [p for p, h, a in zip(pbp['homePlayer6'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]

#etc
date = [d for d, h, a in zip(pbp['Date'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]

events = [e for e, h, a in zip(pbp['Event'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
descriptions = [d for d, h, a in zip(pbp['Description'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
indices = range(len(events))
prev_event = ['crap' if events[i-1]!=events[i-1] or i==0 else events[i-1] for e, i in zip(events, indices)]

home_teams = [h for h, a in zip(pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
away_teams = [a for h, a in zip(pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]

secs = [s for s, h, a in zip(pbp['Seconds_Elapsed'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
since_last = [s-secs[l-1] if l>1 else 0 for s, l in zip(secs, range(len(secs)))]
pers = [p for p, h, a in zip(pbp['Period'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]

str1 = [s for s, h, a in zip(pbp['Strength'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
strength = ['crap' if s!=s else s for s in str1]

z1 = [z for z, h, a in zip(pbp['Ev_Zone'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
zone = ['crap' if z!=z else z for z in z1]

sh1 = [s for s, h, a in zip(pbp['Type'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
shot_type = ['crap' if t!=t or '(' in t else t for t in sh1]

x1 = [x for x, h, a in zip(pbp['xC'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
xcoord = [0 if np.isnan(x)==True else x for x in x1]
y1 = [y for y, h, a in zip(pbp['yC'], pbp['Home_Team'], pbp['Away_Team']) if player_team in a or player_team in h]
ycoord = [0 if np.isnan(y)==True else y for y in y1]

prev_x = np.asarray([0 if np.isnan(xcoord[i-1])==True else xcoord[i-1] for e, i in zip(events, indices)])
prev_y = np.asarray([0 if np.isnan(ycoord[i-1])==True else ycoord[i-1] for e, i in zip(events, indices)])

strength_dict = {'0x5': 0, '5x6': 1, '5x2': 2, '4x4': 3, '5x3': 4, '5x1': 5, '4x3': 6, '4x6': 7, '6x6': 8, '5x0': 9, '4x5': 10, '6x1': 11, '5x7': 12, '5x4': 13, '1x5': 14, '2x5': 15, '3x5': 16, '2x3': 17, '0x0': 18, '1x1': 19, '1x0': 20, '8x5': 21, '-1x0': 22, '0x1': 23, '-1x-1': 24, '6x5': 25, '4x2': 26, '2x4': 27, '3x4': 28, '3x3': 29, '5x5': 30, 'crap':100}
zone_dict = {'Off': 0, 'Neu': 1, 'Def': 2, 'crap':100}
shot_type_dict = {'TIP-IN': 1, 'DEFLECTED': 2, 'WRIST SHOT': 3, 'SLAP SHOT': 4, 'SNAP SHOT': 5, 'BACKHAND': 6, 'WRAP-AROUND': 7, 'crap':100, 'pass':101}
prev_event_dict = {'MISS': 0, 'GIVE': 1, 'CHL': 2, 'SHOT': 3, 'GOAL': 4, 'STOP': 5, 'GEND': 6, 'FAC': 7, 'BLOCK': 8, 'HIT': 9, 'PENL': 10, 'TAKE': 11, 'DELPEN': 12, 'PSTR': 13, 'PEND': 14, 'EISTR': 15, 'crap':100, 'pass':101, 'SOC':102}

strength_nums = [strength_dict[v] if v in strength_dict.values() else 100 for v in strength]
zone_nums = [zone_dict[v] for v in zone]
shot_type_nums = [shot_type_dict[v] for v in shot_type]
prev_event_nums = [prev_event_dict[v] for v in prev_event]

combo = 0
n_events = 0
last_game = 0
combolist = []

for j, e in enumerate(events):

    home_team = home_teams[j]
    away_team = away_teams[j]

    # modifier = 0
    if player_team == home_team:
        modifier = 1
    elif player_team == away_team:
        modifier = -1
    else:
        continue

    on_ice = [a1[j], a2[j], a3[j], a4[j], a5[j], a6[j], h1[j], h2[j], h3[j], h4[j], h5[j], h6[j]]

    if e == 'SHOT' or e == 'GOAL' or e == 'MISS':
        xg = xG(strength_nums[j], xcoord[j], ycoord[j], pers[j], secs[j], since_last[j], zone_nums[j], shot_type_nums[j], prev_event_nums[j], prev_x[j], prev_y[j])
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 0, 1+xg, 0, 0, 0, 0) * modifier
            n_events += 1
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 0, 1+xg, 0, 0, 0, 0) * modifier
            n_events += 1

    elif e == 'BLOCK':
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 1, 0, 0, 0, 0, 0) * modifier
            n_events += 1
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 1, 0, 0, 0, 0, 0) * modifier
            n_events += 1

    elif e == 'PENL':
        p_team = descriptions[j].split(' ')[0]
        mins = int(descriptions[j].split(' min)')[0].split('(')[-1])
        if p_team == home_team:
            combo -= combo_pm(on_ice, player_list, 0, 0, 0, 1, 0, 0) * modifier
            n_events += 1
        elif p_team == away_team:
            combo += combo_pm(on_ice, player_list, 0, 0, 0, 1, 0, 0) * modifier
            n_events += 1

    elif e == 'FAC':
        z, w, wt, l, lt = fo_parser(descriptions[j])
        if wt == home_team:
            combo += combo_pm(on_ice, player_list, 0, 0, 1, 0, 0, 0) * modifier
            n_events += 1
        elif wt == away_team:
            combo -= combo_pm(on_ice, player_list, 0, 0, 1, 0, 0, 0) * modifier
            n_events += 1

    elif e == 'GIVE':
        if descriptions[j].startswith(home_team):
            combo -= combo_pm(on_ice, player_list, 0, 0, 0, 0, 1, 0) * modifier
            n_events += 1
        elif descriptions[j].startswith(away_team):
            combo += combo_pm(on_ice, player_list, 0, 0, 0, 0, 1, 0) * modifier
            n_events += 1

    elif e == 'TAKE':
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 0, 0, 0, 0, 0, 1) * modifier
            n_events += 1
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 0, 0, 0, 0, 0, 1) * modifier
            n_events += 1

    game = date[j]
    if last_game != 0 and last_game != game:
        combolist.append(combo/n_events)
        combo = 0
        n_events = 0
    last_game = game

combolist.append(combo/n_events)

n_games = len([x for x in combolist if x != 0])
total_combo = sum(combolist)

# print(combolist, total_combo)
print(f'{total_combo/n_games:.4f} - {total_combo:.4f} over {n_games:d} games')
