import pandas as pd
import mysql.connector

csv = "NBA-Stat-Compare/nba_2025_all_players_full_season_all_games.csv"

connect = mysql.connector.connect(host = "localhost", user = "root", password = "password123", database = "NBAPlayers")

df = pd.read_csv(csv, low_memory = False)

cursor = connect.cursor()

cursor.execute("DROP TABLE IF EXISTS season2425")

cursor.execute("""
               CREATE TABLE season2425(
               date DATE,
               team VARCHAR(3),
               opponent VARCHAR(3),
               player VARCHAR(35),
               minutes VARCHAR(6),
               FG INT,
               FGA INT,
               FGP DOUBLE,
               THREEP INT,
               THREEPA INT,
               THREEPP DOUBLE,
               FT INT,
               FTA INT,
               FTP DOUBLE,
               ORB INT,
               DRB INT,
               TRB INT,
               AST INT,
               STL INT, 
               BLK INT,
               TOV INT,
               PF INT,
               PTS INT,
               GmSc DOUBLE,
               PlusMinus INT
               )
               """)
# create nothing row
data = []
for row in df.itertuples(index = False, name = None):
    if row[3] != "Reserves":
        if not isinstance(row[5], str) or pd.isna(row[5]):

            data.append(row)

cursor.executemany("""
    INSERT INTO season2425 (
        date, team, opponent, player, minutes,
        FG, FGA, FGP, THREEP, THREEPA, THREEPP, FT, FTA, FTP,
        ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc, PlusMinus
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
""", data)

connect.commit()
cursor.close()
connect.close()









