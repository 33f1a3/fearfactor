import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hockeysuite as hs
import argparse

parser = argparse.ArgumentParser(description='Produce game time plot for advantage')
parser.add_argument('-p', '--pbp', required=True, type=str, help='PBP filename - no extention')
parser.add_argument('-c', '--combo', required=True, type=str, nargs='*', help='List of players - full names')
parser.add_argument('-t', '--team', required=True, type=str, help='3-letter team code for players')

params = parser.parse_args()

pbp_path = params.pbp
player_list = params.combo
player_team = params.team

pbp = pd.read_csv(f'{pbp_path}.csv')

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

def advantager(cf, sf, fw, pd, g, t, s):
    
    return 0.1267380595068711*cf + 0.2797557173984735*sf + 0.08639459801588804*fw + 0.06876239334454985*pd + 0.08165454758042826*g + 0.09950296398800655*t + 0.25719172016578284*s

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

def plusminuser_weighted(on_ice, plus_team, minus_team, cf, sf, fw, pd, g, t, s):
    
    for i in on_ice:
        if i in plus_team.keys():
            plus_team[i] += 1 * advantager(cf, sf, fw, pd, g, t, s)
        elif i in minus_team.keys():
            minus_team[i] -= 1 * advantager(cf, sf, fw, pd, g, t, s)
            
    return(plus_team, minus_team)

def combo_pm(on_ice, players, cf, sf, fw, pd, g, t, s):
    
    if all(p in on_ice for p in players):
        return advantager(cf, sf, fw, pd, g, t, s)
    else:
        return 0

events = pbp['Event']
descriptions = pbp['Description']

combo = 0
last_game = 0
combolist = []

for j, e in enumerate(events):

    home_team = pbp['Home_Team'][j]
    away_team = pbp['Away_Team'][j]

    if player_team == home_team:
        modifier = 1
    elif player_team == away_team:
        modifier = -1
    else:
        continue

    on_ice = [pbp['awayPlayer1'][j], pbp['awayPlayer2'][j], pbp['awayPlayer3'][j], pbp['awayPlayer4'][j], pbp['awayPlayer5'][j], pbp['awayPlayer6'][j], pbp['homePlayer1'][j], pbp['homePlayer2'][j], pbp['homePlayer3'][j], pbp['homePlayer4'][j], pbp['homePlayer5'][j], pbp['homePlayer6'][j]]
        
    if e == 'SHOT':
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 1, 1, 0, 0, 0, 0, 0) * modifier
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 1, 1, 0, 0, 0, 0, 0) * modifier

    elif e == 'GOAL':
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 1, 1, 0, 0, 0, 0, 0) * modifier
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 1, 1, 0, 0, 0, 0, 0) * modifier

    elif e == 'BLOCK' or e == 'MISS':
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 1, 0, 0, 0, 0, 0, 0) * modifier
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 1, 0, 0, 0, 0, 0, 0) * modifier

    elif e == 'PENL':
        p_team = descriptions[j].split(' ')[0]
        mins = int(descriptions[j].split(' min)')[0].split('(')[-1])
        if p_team == home_team:
            combo -= combo_pm(on_ice, player_list, 0, 0, 0, 1, 0, 0, 0) * modifier
        elif p_team == away_team:
            combo += combo_pm(on_ice, player_list, 0, 0, 0, 1, 0, 0, 0) * modifier

    elif e == 'FAC':
        z, w, wt, l, lt = fo_parser(descriptions[j])
        if wt == home_team:
            combo += combo_pm(on_ice, player_list, 0, 0, 1, 0, 0, 0, 0) * modifier
        elif wt == away_team:
            combo -= combo_pm(on_ice, player_list, 0, 0, 1, 0, 0, 0, 0) * modifier

    elif e == 'GIVE':
        if descriptions[j].startswith(home_team):
            combo -= combo_pm(on_ice, player_list, 0, 0, 0, 0, 1, 0, 0) * modifier
        elif descriptions[j].startswith(away_team):
            combo += combo_pm(on_ice, player_list, 0, 0, 0, 0, 1, 0, 0) * modifier

    elif e == 'TAKE':
        if descriptions[j].startswith(home_team):
            combo += combo_pm(on_ice, player_list, 0, 0, 0, 0, 0, 1, 0) * modifier
        elif descriptions[j].startswith(away_team):
            combo -= combo_pm(on_ice, player_list, 0, 0, 0, 0, 0, 1, 0) * modifier

    game = pbp['Date'][j]
    if last_game != 0 and last_game != game:
        combolist.append(combo)
        combo = 0
    last_game = game

combolist.append(combo)

n_games = len([x for x in combolist if x != 0])
total_combo = sum(combolist)

# print(combolist, total_combo)
print(f'{total_combo/n_games:.4f} - {total_combo:.4f} over {n_games:d} games')