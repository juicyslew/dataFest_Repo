import json

json_data=open('../../Data/coord_test.json').read()

data = json.loads(json_data)
print(data['results'][0]['geometry']['location'])
print((data['results'][0]['geometry']['location']['lat'], data['results'][0]['geometry']['location']['lng']))
