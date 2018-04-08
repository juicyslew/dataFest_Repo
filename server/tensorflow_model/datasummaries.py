import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

start = time.time()
df = pd.read_csv("../../Data/datafest_playground.csv", sep=',')
end = time.time()
print ("Loaded data in %fs") % (end - start,)

def histograms(df):
	df['estSalSquare'] = np.log(df['estimatedSalary'] +1)
	df.hist(bins = 600, column = 'estSalSquare')
	plt.show()
	"""for col in df:
		try:
			df.hist(bins = 200, column = col)
			print("plotting %s!") %(col)
			plt.show()
		except:
			print('tried %s, but failed') %(col)
			plt.close()
			continue"""
		#df.plot()
		#plt.figure()

def ScatterPlotSummaries(df):
	#cols_long= ['avgOverallRating', 'numReviews', 'descriptionCharacterLength', 'descriptionWordCount', 'estimatedSalary', 'jobAgeDays', 'clicks', 'localClicks']
	cols_short = ['clicks', 'localClicks', 'descriptionCharacterLength', 'descriptionWordCount']


	for i in range(len(cols_short)):
		for j in range(i+1, len(cols_short)):
			df.plot.scatter(x = cols_short[i], y = cols_short[j])
			plt.show()

def Generate_Metrics(df, num = 8, choices = []):
	"""
	add some randomly generated metrix
	"""
	#new_df = pd.DataFrame()
	for i in range(num):
		#rand1 = random.randint(0,len(df.columns))
		#rand2 = random.randint(0,len(df.columns))
		num1 = random.choice(choices)
		num2 = random.choice(choices)
		col = df.iloc[:, num1]
		col2 = df.iloc[:, num2]
		name = '%s.%s' %(df.columns[num1], df.columns[num2])
		print ('creating %s' %(name))
		values = col * col2
		
		df[name] = values;
	return df



def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
	histograms(df)
	#ScatterPlotSummaries(df);
	#print(df.columns)
	#rand_df = Generate_Metrics(df, num = 6, choices = [8,9,13,14,16,17, 22, 23, 24]); # 25
	#plot_corr(df);
	#plt.matshow(df.corr())
	#plt.show()
	#histograms(df);


### Interesting Things (PERIOD)

# EstimatedSalary vs JobAgeDays **
# EstimatedSalary vs Clicks **
# Clicks vs JobAgeDays **
# newMetric Non-Local-Clicks
# Map of Expected income US.
