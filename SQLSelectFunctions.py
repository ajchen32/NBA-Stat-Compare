import mysql.connector

# GOAL IS TO HAVE CHECKBOX OPTIONS
# A user should be able to modify the dates at any state 
# The goal is for over a user-defined period of time, the program calculates a players "Option number" on the team and the "defensive ranking" 
# of the team that they are playing against in order to predict the score of the player

# "Option Number" - this is local to a players team and is determined by their ppg over a given period relative to their teammates (highest scoring is option 1)
# "Defensive Ranking" - this is of all 32 nba teams over a given time period determined by FG% and modified for 3P%. The modification is for
# increased value of 3 - pointer. Points scored are not directly evaluated because that is also impacted by the pace of a team's offense. 