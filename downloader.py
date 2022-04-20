import hockey_scraper as hs
import argparse

parser = argparse.ArgumentParser(description='Produce game time plot for advantage')
parser.add_argument('-g', '--gameid', required=True, type=int, help='Game ID - see nhl.com/scores')

params = parser.parse_args()

pbp = hs.scrape_games([params.gameid], False, docs_dir='./gamestats/')