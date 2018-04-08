import pickle as pk

with open('Coords.pk', 'rb') as f:
	city_list = pk.load(f)

print(city_list)

print("Length : %d" % len (city_list))
