import pandas as pd
import mysql.connector

csv = ""

connect = mysql.connect(host = "localhost", user = "root", password = "password123", database = "NBAPlayer202425")

df = pd.read_csv(csv)

cursor = connect.cursor()


