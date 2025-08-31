# NBA-Stat-Compare

WIP -> The project implements a simple LSTM seq2seq model to predict future player statlines based on the most recent 5 stat lines.

The goal is for a user to be able to select a player and the opposing team and for the model to return a predicitve statline.


Right now the files work in this order:
1. scrapeStats.py produces "NBA-Stat-Comparenba_2025_games_october_to_june.csv"
2. csvToLinks.py produces "NBA-Stat-Compare/nba_2025_games_with_links.csv"
3. individualStatPuller.py produces "NBA-Stat-Compare/nba_2025_all_players_full_season_all_games.csv"
4. csvtoSQL moves csv data into a locally mysql server
5. dataprep.py produces "NBA-Stat-Compare/trainandtest_nparrays.npz"
6. train.py produces LSTMmodel
