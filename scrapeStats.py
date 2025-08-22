import pandas as pd

months = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]
base_url = "https://www.basketball-reference.com/leagues/NBA_2025_games-{}.html"

all_dfs = []

for month in months:
    url = base_url.format(month)
    
    df = pd.read_html(url)[0]
    all_dfs.append(df)

full_season_df = pd.concat(all_dfs, ignore_index=True)


full_season_df.to_csv("NBA-Stat-Compare/nba_2025_games_october_to_june.csv", index=False)
print("Saved:", "nba_2025_games_october_to_june.csv")