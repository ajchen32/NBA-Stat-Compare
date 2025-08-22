# import pandas as pd
# import time
# import random

# # Mapping of full team names to abbreviations
# team_abbr = {
#     "Boston Celtics": "BOS",
#     "New York Knicks": "NYK",
#     "Golden State Warriors": "GSW",
#     "Oklahoma City Thunder": "OKC",
#     "Detroit Pistons": "DET",
#     "Miami Heat": "MIA",
#     "Los Angeles Lakers": "LAL",
#     "Chicago Bulls": "CHI",
#     "Philadelphia 76ers": "PHI",
#     "Brooklyn Nets": "BRK",
#     "Milwaukee Bucks": "MIL",
#     "Cleveland Cavaliers": "CLE",
#     "Toronto Raptors": "TOR",
#     "Atlanta Hawks": "ATL",
#     "Orlando Magic": "ORL",
#     "Houston Rockets": "HOU",
#     "Utah Jazz": "UTA",
#     "Minnesota Timberwolves": "MIN",
#     "Portland Trail Blazers": "POR",
#     "Sacramento Kings": "SAC",
#     "New Orleans Pelicans": "NOP",
#     "Phoenix Suns": "PHO",
#     "Dallas Mavericks": "DAL",
#     "Memphis Grizzlies": "MEM",
#     "Indiana Pacers": "IND",
#     "Charlotte Hornets": "CHO",
#     "Los Angeles Clippers": "LAC",
#     "Denver Nuggets": "DEN",
#     "Washington Wizards": "WAS",
# }

# def get_first_and_eighth_table_from_boxscore(url, home_team_name, date_str):
#     """
#     Returns first and eighth tables from a Box Score page with Date and Team Abbreviation columns added.
#     """
#     try:
#         tables = pd.read_html(url)
#         if not tables:
#             return None, None

#         abbr = team_abbr.get(home_team_name, "UNK")
        
#         def process_table(df):
#             df = df.iloc[1:-1]  # remove first and last row
#             df.insert(0, "Date", date_str)
#             df.insert(1, "Team", abbr)
#             return df
        
#         first_table = process_table(tables[0])
#         eighth_table = process_table(tables[7]) if len(tables) >= 8 else None
        
#         return first_table, eighth_table

#     except Exception as e:
#         print(f"Error fetching tables from {url}: {e}")
#         return None, None


# # --- Main loop for full season ---
# season_csv = "nba_2025_games_with_links.csv"
# df_season = pd.read_csv(season_csv)

# all_players = []

# for idx, row in df_season.iterrows():
#     url = row['Box_Score_Link']
#     date_str = pd.to_datetime(row['Date']).strftime("%Y-%m-%d")
#     home_team = row['Home/Neutral']
    
#     first_table, eighth_table = get_first_and_eighth_table_from_boxscore(url, home_team, date_str)
    
#     if first_table is not None:
#         all_players.append(first_table)
#     if eighth_table is not None:
#         all_players.append(eighth_table)
    
#     # Wait 2–4 seconds between requests to avoid 429
#     time.sleep(random.uniform(2, 4))

# # Combine all tables into one big DataFrame
# big_df = pd.concat(all_players, ignore_index=True)

# # Save as a single CSV
# big_df.to_csv("nba_2025_all_players_full_season.csv", index=False)
# print("Saved full season player data to nba_2025_all_players_full_season.csv")
import pandas as pd
import time
import random

# Example mapping of team full names to abbreviations
team_abbr = {
    "Boston Celtics": "BOS",
    "New York Knicks": "NYK",
    "Golden State Warriors": "GSW",
    "Oklahoma City Thunder": "OKC",
    "Detroit Pistons": "DET",
    "Miami Heat": "MIA",
    # ... add all teams here
}

def get_first_and_eighth_table_from_boxscore(url, home_team_name):
    """
    Returns first and eighth tables with Date and Team Abbreviation columns added.
    """
    try:
        tables = pd.read_html(url)
        if not tables:
            return None, None
        
        date_str = pd.to_datetime(pd.Timestamp.now()).strftime("%Y-%m-%d")  # placeholder, will set properly later
        abbr = team_abbr.get(home_team_name, "UNK")
        
        def process_table(df):
            df = df.iloc[1:-1]  # remove first and last row
            df.insert(0, "Date", date_str)          # first column
            df.insert(1, "Team", abbr)             # second column
            return df
        
        first_table = process_table(tables[0])
        eighth_table = process_table(tables[7]) if len(tables) >= 8 else None
        
        return first_table, eighth_table

    except Exception as e:
        print(f"Error fetching tables from {url}: {e}")
        return None, None


# --- Main loop for a test batch ---
season_csv = "NBA-Stat-Compare/nba_2025_games_with_links.csv"
df_season = pd.read_csv(season_csv)

all_players = []

for idx, row in df_season.head(10).iterrows():  # example: first 10 games
    url = row['Box_Score_Link']
    date_str = pd.to_datetime(row['Date']).strftime("%Y-%m-%d")
    home_team = row['Home/Neutral']
    
    first_table, eighth_table = get_first_and_eighth_table_from_boxscore(url, home_team)
    
    # Replace placeholder date with actual game date
    if first_table is not None:
        first_table['Date'] = date_str
        all_players.append(first_table)
    if eighth_table is not None:
        eighth_table['Date'] = date_str
        all_players.append(eighth_table)
    
    time.sleep(random.uniform(2, 4))  # delay to avoid 429

# Combine all tables into one DataFrame
big_df = pd.concat(all_players, ignore_index=True)

# Save as a single CSV
big_df.to_csv("nba_2025_all_players_first10_with_team.csv", index=False)
print("Saved first 10 games to nba_2025_all_players_first10_with_team.csv")
