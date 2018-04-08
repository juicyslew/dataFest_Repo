import pandas as pd
import numpy as np
import pickle as pk
import math

df = pd.read_csv("../../Data/final.csv", sep=',')
with open('Coords.pk', 'rb') as f:
	city_list = pk.load(f)

#df = df.loc[df['country'] == 'US']
#df = df[pd.notnull(df['stateProvince'])]
#df = df[pd.notnull(df['city'])]
df['cityState'] = df["city"].map(str) + ', ' + df["stateProvince"]
df = df.drop_duplicates('cityState')

city_lat = dict()
city_lng = dict()
#print (city_list)

#create lat and lng map
for entry in city_list:
	print(entry)
	if (entry == None):
		continue
	city_lat[entry[0]] = entry[1][0]
	city_lng[entry[0]] = entry[1][1]

#create lat and lng columns
df['lat'] = df['cityState'].map(city_lat)
df['lng'] = df['cityState'].map(city_lng)

#remove nans
df = df[pd.notnull(df['lat'])]
df = df[pd.notnull(df['lng'])]

df.to_csv("../../Data/final_latlng.csv", sep=',')
