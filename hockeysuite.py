import pandas as pd
import numpy as np

def get_team(toi_table, pbp_table, team):
    """
    Get a list of the players in a given team over a range of games
    
    Args:
        toi_table (Pandas DataFrame): input data, all TOI
        team (string): team code, e.g. PHI
        
    Returns:
        players (list): list of player surname strings
        positions (list): list of player position strings
    """
    
    many_players = [p for p, t in zip(toi_table['Player'], toi_table['Team']) if t==team]
    full_names = list(set(many_players))
    players = [n.split(' ')[-1] for n in full_names]
    player_ids = [get_id(toi_table, p, team) for p in players]
    positions = [get_position(pbp_table, i) for p, i in zip(players, player_ids)]

    return players, positions

def get_id(toi_table, player, team):
    """
    Get a player's ID
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()

    player_id = [i for i, p, t in zip(toi_table['Player_Id'], toi_table['Player'], toi_table['Team']) if player in p and t==team][0]
    
    return player_id

def get_position(pbp_table, player_id):
    """
    Get a player's position

    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        playerId (integer): player ID

    Returns:
        position (string): 'D', 'F', or 'G'
    """

    defense1 = [d for d, s in zip(pbp_table['awayPlayer4_id'], pbp_table['Strength']) if s == '5x5']
    defense2 = [d for d, s in zip(pbp_table['awayPlayer5_id'], pbp_table['Strength']) if s == '5x5']
    defense3 = [d for d, s in zip(pbp_table['homePlayer4_id'], pbp_table['Strength']) if s == '5x5']
    defense4 = [d for d, s in zip(pbp_table['homePlayer5_id'], pbp_table['Strength']) if s == '5x5']
    defenders = [*defense1, *defense2, *defense3, *defense4]
    d_count = defenders.count(player_id)

    forward1 = [f for f, s in zip(pbp_table['awayPlayer1_id'], pbp_table['Strength']) if s == '5x5']
    forward2 = [f for f, s in zip(pbp_table['awayPlayer2_id'], pbp_table['Strength']) if s == '5x5']
    forward3 = [f for f, s in zip(pbp_table['awayPlayer3_id'], pbp_table['Strength']) if s == '5x5']
    forward4 = [f for f, s in zip(pbp_table['homePlayer1_id'], pbp_table['Strength']) if s == '5x5']
    forward5 = [f for f, s in zip(pbp_table['homePlayer2_id'], pbp_table['Strength']) if s == '5x5']
    forward6 = [f for f, s in zip(pbp_table['homePlayer3_id'], pbp_table['Strength']) if s == '5x5']
    forwards = [*forward1, *forward2, *forward3, *forward4, *forward5, *forward6]
    f_count = forwards.count(player_id)

    goalie1 = [g for g, s in zip(pbp_table['awayPlayer6_id'], pbp_table['Strength']) if s == '5x5']
    goalie2 = [g for g, s in zip(pbp_table['homePlayer6_id'], pbp_table['Strength']) if s == '5x5']
    goalies = [*goalie1, *goalie2]
    g_count = goalies.count(player_id)

    if g_count > d_count and g_count > f_count:
        position = 'G'
    elif d_count > f_count:
        position = 'D'
    else:
        position = 'F'

    return position

def get_toi(toi_table, player, team):
    """
    Find a player's cumulative TOI and average TOI
    
    Args:
        toi_table (Pandas DataFrame): input data, all TOI
        player (string): player's name (not case-sensitive)
        team (string): three-character team code (case-sensitive)
    
    Returns:
        toi (float): total ice time in minutes
        avg_toi (float): average ice time in minutes
        games_played (int): number of games played
    """
    
    player_id = get_id(toi_table, player, team)
    
    tois = [d for d, i, t in zip(toi_table['Duration'], toi_table['Player_Id'], toi_table['Team']) if i==player_id and t==team]
    game_ids = [g for g, i, t in zip(toi_table['Game_Id'], toi_table['Player_Id'], toi_table['Team']) if i==player_id and t==team]
    toi = sum(tois)/60
    games_played = len(list(set(game_ids)))
    avg_toi = toi / games_played
    
    return toi, avg_toi, games_played

def shot_parse(desc):
    """
    Parse shot event descriptions: includes goals, shots, blocks, and misses
    
    Args:
        desc (string): individual event description
        
    Returns:
        player who does the action
        their team
        
    """
    
    # SHOT
    if 'ONGOAL' in desc:
        parts = desc.split(', ')
        team = parts[0].split(' ')[0]
        player = parts[0].split(' ')[-1]
        shot_type = parts[1]
        # zone = parts[2].split('.')[0]
        try:
            ft_away = parts[3].split(' ')[0]
        except IndexError:
            ft_away = None
        blocker = None
        
    # BLOCK
    elif 'BLOCKED' in desc:
        parts = desc.split(', ')
        team = parts[0].split(' ')[0]
        player = parts[0].split(' ')[2]
        shot_type = parts[1]
        # zone = parts[2].split('.')[0]
        try:
            ft_away = None
        except IndexError:
            ft_away = None
        blocker = parts[0].split(' ')[-1]
        
    # MISS
    elif 'Wide' in desc:
        parts = desc.split(', ')
        team = parts[0].split(' ')[0]
        player = parts[0].split(' ')[-1]
        shot_type = parts[1]
        # zone = parts[3].split('.')[0]
        try:
            ft_away = parts[4].split(' ')[0]
        except IndexError:
            ft_away = None
        blocker = None

    # GOAL
    elif 'Assist' in desc:
        parts = desc.split(', ')
        team = parts[0].split(' ')[0]
        player = parts[0].split(' ')[-1].split('(')[0]
        shot_type = parts[1]
        # zone = parts[2].split('.')[0]
        try:
            ft_away = parts[3].split(' ')[0]
        except IndexError:
            ft_away = None
        blocker = None
        
    # catch faceoffs etc
    else:
        team = None
        player = None
        shot_type = None
        # zone = None
        ft_away = None
        blocker = None     
        
    return team, player, shot_type, ft_away, blocker

def get_corsi(pbp_table, player, team):
    """
    Get the cumulative Corsi for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): final word of player surname, e.g. 'BRAUN' or 'RIEMSDYK'
        
    Returns:
        corsi (int): player's total Corsi for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc in pbp_table['Description'] if player in desc]
    corsi = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_player == player and shot_team == team:
            corsi += 1
    
    return corsi

def get_ev_corsi(pbp_table, player, team):
    """
    Get the cumulative Corsi for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): final word of player surname, e.g. 'BRAUN' or 'RIEMSDYK'
        
    Returns:
        corsi (int): player's total Corsi for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc, s in zip(pbp_table['Description'], pbp_table['Strength']) if player in desc and s=='5x5']
    corsi = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_player == player and shot_team == team:
            corsi += 1
    
    return corsi

def get_fenwick(pbp_table, player, team):
    """
    Get the cumulative Fenwick for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): player name (not case-sensitive)
        
    Returns:
        fenwick (int): player's total Fenwick for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc in pbp_table['Description'] if player in desc]
    fenwick = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_player == player and blocker == None and shot_team == team:
            fenwick += 1
    
    return fenwick

def get_misses(pbp_table, player, team):
    """
    Get the number of missed shots for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): player name (not case-sensitive)
        
    Returns:
        misses (int): player's total missed shots for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc, event in zip(pbp_table['Description'], pbp_table['Event']) if player in desc and event=='MISS']
    misses = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_team == team:
            misses += 1
    
    return misses

def get_blocks(pbp_table, player, team):
    """
    Get the number of blocked shots for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): player name (not case-sensitive)
        
    Returns:
        blocks (int): player's total blocked shots for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc, event in zip(pbp_table['Description'], pbp_table['Event']) if player in desc and event=='BLOCK']
    blocks = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_team == team:
            blocks += 1
    
    return blocks

def get_saves(pbp_table, player, team):
    """
    Get the number of saved/deflected shots for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): player name (not case-sensitive)
        
    Returns:
        saves (int): player's total saved shots for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc, event in zip(pbp_table['Description'], pbp_table['Event']) if player in desc and event=='SHOT']
    saves = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_team == team:
            saves += 1
    
    return saves

def get_points(pbp_table, player, team):
    """
    Get the number of points for one player over a range of games
    
    Args:
        pbp_table (Pandas DataFrame): input data, all game events
        player (string): player name (not case-sensitive)
        
    Returns:
        points (int): player's total points for the input games
    """

    if ' ' in player: # if passed full name, just use surname to grab
        player = player.split(' ')[-1]
    player = player.upper()
    
    unfiltered_shots = [desc for desc, event in zip(pbp_table['Description'], pbp_table['Event']) if player in desc and event=='GOAL']
    points = 0
    for shot in unfiltered_shots:
        shot_team, shot_player, shot_type, ft_away, blocker = shot_parse(shot)
        if shot_team == team:
            points += 1
    
    return points

def get_apnm(pbp_table, toi_table, player, team):
    """
    All Plus No Minus: every time a player is on ice for a Corsi shot
    Rough measure of assists
    """

    player_id = get_id(toi_table, player, team)
    
    events = ['SHOT', 'MISS', 'BLOCK', 'GOAL']
    on_ice = [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, e, d in zip(pbp_table['awayPlayer1_id'], pbp_table['awayPlayer2_id'], pbp_table['awayPlayer3_id'], pbp_table['awayPlayer4_id'], pbp_table['awayPlayer5_id'], pbp_table['awayPlayer6_id'], pbp_table['homePlayer1_id'], pbp_table['homePlayer2_id'], pbp_table['homePlayer3_id'], pbp_table['homePlayer4_id'], pbp_table['homePlayer5_id'], pbp_table['homePlayer6_id'], pbp_table['Event'], pbp_table['Description']) if e in events and d.startswith(team)==True]
    apnm = 0
    for i in range(len(on_ice)):
        if player_id in on_ice[i]:
            apnm += 1
    
    return apnm

def get_ev_apnm(pbp_table, toi_table, player, team):
    """
    All Plus No Minus: every time a player is on ice for a Corsi shot at even strength
    Rough measure of assists
    """
    
    player_id = get_id(toi_table, player, team)
    
    events = ['SHOT', 'MISS', 'BLOCK', 'GOAL']
    on_ice = [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, e, d, s in zip(pbp_table['awayPlayer1_id'], pbp_table['awayPlayer2_id'], pbp_table['awayPlayer3_id'], pbp_table['awayPlayer4_id'], pbp_table['awayPlayer5_id'], pbp_table['awayPlayer6_id'], pbp_table['homePlayer1_id'], pbp_table['homePlayer2_id'], pbp_table['homePlayer3_id'], pbp_table['homePlayer4_id'], pbp_table['homePlayer5_id'], pbp_table['homePlayer6_id'], pbp_table['Event'], pbp_table['Description'], pbp_table['Strength']) if e in events and d.startswith(team)==True and s=='5x5']
    apnm = 0
    for i in range(len(on_ice)):
        if player_id in on_ice[i]:
            apnm += 1
    
    return apnm

def get_apnm_without(pbp_table, toi_table, player, player2, team):
    """
    All Plus No Minus: every time a player is on ice for a Corsi shot
    Rough measure of assists
    """

    player_id = get_id(toi_table, player, team)
    player_id2 = get_id(toi_table, player2, team)
    
    events = ['SHOT', 'MISS', 'BLOCK', 'GOAL']
    on_ice = [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, e, d in zip(pbp_table['awayPlayer1_id'], pbp_table['awayPlayer2_id'], pbp_table['awayPlayer3_id'], pbp_table['awayPlayer4_id'], pbp_table['awayPlayer5_id'], pbp_table['awayPlayer6_id'], pbp_table['homePlayer1_id'], pbp_table['homePlayer2_id'], pbp_table['homePlayer3_id'], pbp_table['homePlayer4_id'], pbp_table['homePlayer5_id'], pbp_table['homePlayer6_id'], pbp_table['Event'], pbp_table['Description']) if e in events and d.startswith(team)==True]
    apnm = 0
    for i in range(len(on_ice)):
        if player_id in on_ice[i] and player_id2 not in on_ice[i]:
            apnm += 1
    
    return apnm

def get_apnm_with(pbp_table, toi_table, player, player2, team):
    """
    All Plus No Minus: every time a player is on ice for a Corsi shot
    Rough measure of assists
    """

    player_id = get_id(toi_table, player, team)
    player_id2 = get_id(toi_table, player2, team)
    
    events = ['SHOT', 'MISS', 'BLOCK', 'GOAL']
    on_ice = [[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, e, d in zip(pbp_table['awayPlayer1_id'], pbp_table['awayPlayer2_id'], pbp_table['awayPlayer3_id'], pbp_table['awayPlayer4_id'], pbp_table['awayPlayer5_id'], pbp_table['awayPlayer6_id'], pbp_table['homePlayer1_id'], pbp_table['homePlayer2_id'], pbp_table['homePlayer3_id'], pbp_table['homePlayer4_id'], pbp_table['homePlayer5_id'], pbp_table['homePlayer6_id'], pbp_table['Event'], pbp_table['Description']) if e in events and d.startswith(team)==True]
    apnm = 0
    for i in range(len(on_ice)):
        if player_id in on_ice[i] and player_id2 in on_ice[i]:
            apnm += 1
    
    return apnm

def event_parse(event, description, pbp_table):
    """
    Parse event description string
    """

    output_names = ['goals', 'sog', 'fenwick', 'corsi', 'gives', 'takes']
    outputs = np.zeros(len(output_names))
    team = description[:3]

    if event == 'SHOT':
        outputs[1] += 1
        outputs[2] += 1
        outputs[3] += 1
        player = description.split(', ')[0].split(' ')[-1]
        
    elif event == 'GOAL':
        outputs[0] += 1
        outputs[1] += 1
        outputs[2] += 1
        outputs[3] += 1
        player = description.split('(')[0].split(' ')[-1]
            
    elif event == 'BLOCK':
        outputs[2] += 1
        outputs[3] += 1
        player = description.split('BLOCKED')[0].split(' ')[-1]

    elif event == 'MISS':
        outputs[3] += 1
        player = description.split(', ')[0].split(' ')[-1]
            
    elif event == 'GIVE':
        outputs[4] += 1
        player = description.split(', ')[0].split(' ')[-1]
                
    elif event == 'TAKE':
        outputs[5] += 1
        player = description.split(', ')[0].split(' ')[-1]

    else:
        player = None


    # elif e == 'PENL':
    #     p_team = descriptions[j].split(' ')[0]
    #     mins = int(descriptions[j].split(' min)')[0].split('(')[-1])
    #     if p_team == home_team:
    #         pmins_drawn_away[i] += mins
    #     elif p_team == away_team:
    #         pmins_drawn_home[i] += mins
            
    # elif e == 'FAC':
    #     z, w, wt, l, lt = fo_parser(descriptions[j])
    #     if wt == home_team:
    #         f_home[i] += 1
    #     elif wt == away_team:
    #         f_away[i] += 1

    return player, team, outputs

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