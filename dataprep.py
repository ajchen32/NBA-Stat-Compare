import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
from datetime import date

divide_date = date(2025, 3, 15)
divide_date_str = divide_date.strftime('%Y-%m-%d')

# initialize cursor
connect = mysql.connector.connect(host = "localhost", user = "root", password = "password123", database = "NBAPlayers")
cursor = connect.cursor()

# create pandas df with all players sorted by name and date
train_query  = "SELECT date, opponent, player, FG, FGA, FGP, THREEP, THREEPA, THREEPP, FT, FTA, FTP, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc, PlusMinus FROM season2425 WHERE date < %s ORDER BY player, date"
cursor.execute(train_query, (divide_date_str,) )
fetch = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
train_df = pd.DataFrame(fetch, columns = column_names)

test_query = "SELECT date, opponent, player, FG, FGA, FGP, THREEP, THREEPA, THREEPP, FT, FTA, FTP, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc, PlusMinus FROM season2425 AS s1 WHERE date >= %s AND EXISTS(SELECT DISTINCT s2.player FROM season2425 AS s2 WHERE s2.date < %s AND s2.player = s1.player) ORDER BY player, date"
cursor.execute(test_query, (divide_date_str,divide_date_str) )
fetch = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
test_df = pd.DataFrame(fetch, columns = column_names)
# before dummies, create a template on how to form data

# This is from when all data was pulled in one query
# df['date'] = pd.to_datetime(df['date'])
# count_training_rows = (df['date'] <= divide_date_str).sum()
# count_testing_rows = (df['date'] > divide_date_str).sum()


# split up train and test
# this will ensure that the amount of games per player is mod 6 = 0 so that the x vals can be made
# this first iteration is for the training dataset
current_iter = len(train_df) - 1
while(current_iter >= 0):
    if current_iter - 5 >= 0 and train_df.at[current_iter, 'player'] == train_df.at[current_iter - 5, 'player']:
        current_iter -= 6
    else:
        # if the amount cannot reach up to 5, 
        curr_name = train_df.at[current_iter, 'player']
        for i in range(6):
            if current_iter - i < 0 or train_df.at[current_iter - i, 'player'] != curr_name:
                current_iter = current_iter -i
                break
            else:
                train_df.drop(current_iter-i, inplace=True)


current_iter = len(test_df) - 1
# test_df.reset_index(drop=True, inplace=True)
while(current_iter >= 0):
    
    if current_iter - 5 >= 0 and test_df.at[current_iter, 'player'] == test_df.at[current_iter - 5, 'player']:
        current_iter -= 6
    else:
        # if the amount cannot reach up to 5, 
        curr_name = test_df.at[current_iter, 'player']
        for i in range(6):
            if current_iter - i < 0 or test_df.at[current_iter - i, 'player'] != curr_name:
                current_iter = current_iter -i
                break
            else:
                test_df.drop(current_iter-i, inplace=True)




# after processing rows, add a column to the training data with future opponent:
# this adds a column 
#print(len(train_df) % 6) # both equal 0
#print(len(test_df)%6) # both equal 0
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_df['future-opponent'] = None # this denotes who the player will play in the future on theri "6th game" in the y-data
for i in range(len(train_df)):
    if not ((i - 5) % 6 == 0):
        train_df.at[i, 'future-opponent'] = train_df.at[i + 5 - (i%6), 'opponent']
        

test_df['future-opponent'] = None
for i in range(len(test_df)):
    if not ((i - 5) % 6 == 0):
        test_df.at[i, 'future-opponent'] = test_df.at[i + 5 - (i%6), 'opponent']



# need a train_x, train_y, test_x, test_y
# iterate through whole df and make sure that each player has %6 games, 
# 5 of these games are x val, and 1 is the y val

# one hot encode the player names
traindf_encoded = pd.get_dummies(train_df, columns=['opponent', 'player', 'future-opponent'], drop_first =True)
saved_onehot_cols = traindf_encoded.columns.tolist()
pd.Series(saved_onehot_cols).to_csv("NBA-Stat-Compare/team_columns.csv", index = False, header = False)

testdf_encoded = pd.get_dummies(test_df, columns = ['opponent', 'player', 'future-opponent'], drop_first = True).reindex(columns=traindf_encoded.columns, fill_value=0)

# normalize the non-binary stats
scaler = MinMaxScaler()
traindf_encoded.drop(columns=['date'], inplace=True)
testdf_encoded.drop(columns = ['date'], inplace = True)
traindf_numpy = scaler.fit_transform(traindf_encoded)
testdf_numpy = scaler.fit_transform(testdf_encoded)

train_x = None
train_y = None
test_x = None
test_y = None

for i in range(0,len(traindf_numpy),6):
    if train_x is None:
        train_x = traindf_numpy[0:6].copy()
        train_y = traindf_numpy[6].copy()
    else:
        train_x = np.vstack((train_x, traindf_numpy[i:i+5]))
        train_y = np.vstack((train_y, traindf_numpy[i+5]))
    
for i in range(0,len(testdf_numpy),6):
    if test_x is None:
        test_x = testdf_numpy[0:6].copy()
        test_y = testdf_numpy[6].copy()
    else:
        test_x = np.vstack((test_x, testdf_numpy[i:i+5]))
        test_y = np.vstack((test_y, testdf_numpy[i+5]))

# save the numpy arrays for future use::::
np.savez("NBA-Stat-Compare/trainandtest_nparrays.npz", train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y )

