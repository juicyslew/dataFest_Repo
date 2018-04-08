import pandas as pd
import numpy as np
import time
import pickle as pk
import math
import json
import urllib2
import multiprocessing as mp
from contextlib import closing
import os
from itertools import cycle
from collections import defaultdict
from sets import Set

APIiter = cycle(['AIzaSyA0XNhZaQNVsItWIH8bzzs8Zy6RYtmBxUU',
	'AIzaSyArqNY1F9I6VyDCP51F4Z6hNqHpcOyKYQ4',
	'AIzaSyAkPgXQF0zOpc0P5RpvP8zMlgfisdOcxIA',
	'AIzaSyA_DSH87XuuV092P5W8SaIt9zRup2XjfMM',
	'AIzaSyC3HNps5O58m7Kgxj2GLuMxAHdjlay8cqk',
	'AIzaSyDZ55VQhYq2T-7u98ccniotl_Zo9ya-8fA',
	'AIzaSyCpx1bBABlFAiPJmYZfYshgmuEELkAjTfM',
	'AIzaSyDC7Zkaqrya00FNIruRZSC1dBpto5_-odY',
	'AIzaSyD-Iy7gibL3NbDVGt3VOrfJ_vVCZK6tpf4',
	'AIzaSyBKMKhfjJvMKH5t4dagleyk_F3tpsTcUtA',
	'AIzaSyDhSxHFXqUnLCAULKCT9ysn3beOTa6-FKA', 
	'AIzaSyC3I_3I2s0x5MTn9B8k8NvadOCIHQHVUPs', 
	'AIzaSyAcmfoQcIsf57j5QxIyRsxgxMz3glg78z0', 
	'AIzaSyDR0l28zJ_EDynn3hePenljLlnzxu1F6Zo', 
	'AIzaSyAFLGDUdohaQA6CuzNOaREjJKXdYOcXz7o', 
	'AIzaSyAx5aO73ICxRiwRi-xuebA5YV5byuoA9Tg', 
	'AIzaSyCkUpTkQ0rU5azgcANPGzRFbtfYJC8mVqo',
	'AIzaSyBIftfzTBeCIiT7pWQS8waWPXwTEbKjhMk',
	'AIzaSyAyekm1f3bgjzpXkMYDZraCVF_v-71DPJI',
	'AIzaSyDMwppDhZ-HgQGc4c71-515RCYKrbHqDoc',
	'AIzaSyA0XNhZaQNVsItWIH8bzzs8Zy6RYtmBxUU'])


''''''
http_loc = 'https://maps.googleapis.com/maps/api/geocode/json?address='
key_line = '&key='

dump_file = 'Coords.pk'
missed_file = 'missed.pk'


df = pd.read_csv("../Data/datafest_playground.csv", sep=',')

df = df.loc[df['country'] == 'US']
df = df[pd.notnull(df['stateProvince'])]
df = df[pd.notnull(df['city'])]
df['cityState'] = df["city"].map(str) + ', ' + df["stateProvince"]
df = df.drop_duplicates('cityState')


#Multithreading
#output = mp.Queue()



#df = df.loc[df['city'] != float('nan')];
#print(len(APIlist))

#city_dict = {}
found_cities = set()

"""for index, row in df.iterrows():
	if (type(row['cityState']) != str):
		print(type(row['cityState']))"""

def CityConversion(in_val):
	row = in_val[1]
	if (isinstance(row['cityState'], str) and not row['cityState'] in found_cities):
		search_address = (row['cityState']).replace(' ', '+')
		
		lock.acquire()
		API_key = APIiter.next()
		lock.release()

		search_http = http_loc + search_address + key_line + API_key
		try:
			url = urllib2.urlopen(search_http)
		except:
			print ('url error')
			#return None #(row['cityState'], -1)

		json_data= json.loads(url.read())

		if (json_data['status'] == 'OK'):
			lat_lng = (json_data['results'][0]['geometry']['location']['lat'], json_data['results'][0]['geometry']['location']['lng'])
			print(lat_lng)
			#city_dict[row['cityState']] = lat_lng #http_loc + row['city'] + key_line + API_key ; #print ('row %i: %s' %(index, row['city']))#city_dict[row['city']] = http
			if not isinstance((row['cityState'], lat_lng), tuple):
				print('WHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT')
			found_cities.add(row['cityState'])
			return row['cityState'], lat_lng
		else:
			print (json_data['status'])
			print('missed %s' %row['cityState'])
			#return None #(row['cityState'], -1)
			#missed_Cities.append(row['cityState'])
	#return (-1, -1)
	#return search_http[0]

"""i = 0;
for index, row in df.iterrows():
	if (isinstance(row['cityState'], float) and math.isnan(row['cityState'])):
		continue
	search_address = (row['cityState']).replace(' ', '+')
	API_ind = (i+6) % len(APIlist)
	search_http = http_loc + search_address + key_line + APIlist[API_ind]
	print(search_http)
	start = time.time()
	url = urllib2.urlopen(search_http);
	readed = url.read()
	json_data = json.loads(readed)

	if (json_data['status'] == 'OK'):
		lat_lng = index#(json_data['results'][0]['geometry']['location']['lat'], json_data['results'][0]['geometry']['location']['lng'])
		print(lat_lng)
		city_dict[row['cityState']] = lat_lng #http_loc + row['city'] + key_line + API_key ; #print ('row %i: %s' %(index, row['city']))#city_dict[row['city']] = http
	else:
		missed_Cities.append(row['cityState'])
	i += 1;
	if (i >= 2):
		break

with open(dump_file, 'wb') as f:
	pk.dump(city_dict, f)
with open(missed_file, 'wb') as f:
	pk.dump(missed_file, f)

print("Length : %d" % len (city_dict))"""



#random.seed(123)

# Define an output queue


# define a example function
"""def rand_string(length, output):
    # Generates a random string of numbers, lower- and uppercase chars. 
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    output.put(rand_str)"""

def init(l):
    global lock
    lock = l

"""def f(x):
	print(APIlist.next())
	return x*x;"""

if __name__ == '__main__':
    # start 4 worker processes
    l = mp.Lock()
    p_pool = mp.Pool(processes=mp.cpu_count()*2, initializer=init, initargs=(l,))
#    p_results = p_pool.map(urllib2.urlopen, urls)
    p_results = p_pool.map(CityConversion, df.iterrows())
    #for cityState, coords in (r for r in p_results if r is not None)
    #result = defaultdict()
    p_pool.close()
    p_pool.join()
#with closing(mp.Pool(processes=4, initializer=init, initargs=(l,))) as pool:
	#lock = mp.Lock()

    # print "[0, 1, 4,..., 81]"
    #print(pool.map(CityConversion, ))

    # print same numbers in arbitrary order
    #for i in pool.imap_unordered(CityConversion, ):
    #    print(i)

    # evaluate "f(20)" asynchronously
    #res = pool.apply_async(f, (20,))      # runs in *only* one process
    #print(res.get(timeout=1))             # prints "400"

    # evaluate "os.getpid()" asynchronously
    #res = pool.apply_async(os.getpid, ()) # runs in *only* one process
    #print(res.get(timeout=1))             # prints the PID of that process

    # launching multiple evaluations asynchronously *may* use more processes
    """multiple_results = [pool.apply_async(CityConversion, (row,)) for index, row in df.iterrows()]
    print('coagulating processes')
    city_list = [res.get(timeout=10000) for res in multiple_results]"""

    with open(dump_file, 'wb') as f:
    	pk.dump(p_results, f)
	#with open(missed_file, 'wb') as f:
	#	pk.dump(missed_Cities, f)

        # make a single worker sleep for 10 secs
        #res = pool.apply_async(time.sleep, (10,))
        #try:
        #    print(res.get(timeout=1))
        #except TimeoutError:
        #    print("We lacked patience and got a multiprocessing.TimeoutError")

        #print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
