import pandas as pd
import mysql.connector

csv = ""

connect = mysql.connector(host = "localhost", user = "root", password = "password123", database = "NBAPlayers")

df = pd.read_csv(csv)

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
               3P INT,
               3PA INT,
               3PP DOUBLE,
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
data = [tuple(pd.isna(value) for value in row) for row in df.itertuples(index = False, name = None)]

cursor.executemany("""
    INSERT INTO season2425 (
        date, team, opponent, player, minutes,
        FG, FGA, FGP, 3P, 3PA, 3PP, FT, FTA, FTP,
        ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc, PlusMinus
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
""", data)

connect.commit()
cursor.close()
connect.close()







# You need to convert minutes to double from string

