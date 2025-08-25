import pandas as pd
import time
import random

# Mapping of full team names to abbreviations
team_abbr = {
    "Boston Celtics": "BOS",
    "New York Knicks": "NYK",
    "Golden State Warriors": "GSW",
    "Oklahoma City Thunder": "OKC",
    "Detroit Pistons": "DET",
    "Miami Heat": "MIA",
    "Los Angeles Lakers": "LAL",
    "Chicago Bulls": "CHI",
    "Philadelphia 76ers": "PHI",
    "Brooklyn Nets": "BRK",
    "Milwaukee Bucks": "MIL",
    "Cleveland Cavaliers": "CLE",
    "Toronto Raptors": "TOR",
    "Atlanta Hawks": "ATL",
    "Orlando Magic": "ORL",
    "Houston Rockets": "HOU",
    "Utah Jazz": "UTA",
    "Minnesota Timberwolves": "MIN",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "New Orleans Pelicans": "NOP",
    "Phoenix Suns": "PHO",
    "Dallas Mavericks": "DAL",
    "Memphis Grizzlies": "MEM",
    "Indiana Pacers": "IND",
    "Charlotte Hornets": "CHO",
    "Los Angeles Clippers": "LAC",
    "Denver Nuggets": "DEN",
    "Washington Wizards": "WAS",
}

def get_team_tables(url, home_team_name, visitor_team_name, date_str):
    """
    Returns both player tables with Date, Team, and Opponent.
    """
    try:
        tables = pd.read_html(url)
        if not tables:
            return None, None
        
        def process_table(df, team_name, opp_name):
            abbr = team_abbr.get(team_name, "UNK")
            opp_abbr = team_abbr.get(opp_name, "UNK")
            # drop first row (header duplicate) and last row (team totals)
            df = df.iloc[1:-1]
            df.insert(0, "Date", date_str)
            df.insert(1, "Team", abbr)
            df.insert(2, "Opponent", opp_abbr)
            return df

        away_table = process_table(tables[0], visitor_team_name, home_team_name)
        home_table = process_table(tables[8], home_team_name, visitor_team_name) if len(tables) >= 9 else None
        
        return away_table, home_table

    except Exception as e:
        print(f"Error fetching tables from {url}: {e}")
        return None, None


# --- Main loop for full season ---
season_csv = "NBA-Stat-Compare/nba_2025_games_with_links.csv"
df_season = pd.read_csv(season_csv)

all_players = []

for idx, row in df_season.iterrows():
    url = row['Box_Score_Link']
    date_str = pd.to_datetime(row['Date']).strftime("%Y-%m-%d")
    home_team = row['Home/Neutral']
    visitor_team = row['Visitor/Neutral']
    
    away_table, home_table = get_team_tables(url, home_team, visitor_team, date_str)
    
    if away_table is not None:
        all_players.append(away_table)
    if home_table is not None:
        all_players.append(home_table)
    
    # pause between requests to avoid being blocked
    time.sleep(random.uniform(2, 4))

# Combine into single DataFrame
big_df = pd.concat(all_players, ignore_index=True)

# Save as CSV
output_file = "nba_2025_all_players_full_season_with_team_opp.csv"
big_df.to_csv(output_file, index=False)

print(f"Saved full season player data to {output_file}")
