import tensorflow as tf
import pandas as pd
import numpy as np
import pickle as pk

class NN_Model:
	def __init__(self):
		self.sess = tf.Session() 
		saver = tf.train.import_meta_graph('mlModel/model.meta')
		saver.restore(self.sess,tf.train.latest_checkpoint('mlModel/'))

		self.graph = tf.get_default_graph()
		#self.ansmeans = graph.get_tensor_by_name('ansmeans')
		#self.ansstands = graph.get_tensor_by_name('ansstands')

		self.dropout_rate = self.graph.get_tensor_by_name("dropout_rate:0")
		#assign_op = dropout_rate.assign(1)

		self.X = self.graph.get_tensor_by_name("X:0")
		self.y = self.graph.get_tensor_by_name("y:0")
		
		with open('continuous_list.pk', 'rb') as f:
			self.cont_list = pk.load(f);

		with open('cat_dict.pk', 'rb') as f:
			self.cat_dicts = pk.load(f);
		#feed_dict ={X:13.0,y:17.0}
		 
		#Now, access the op that you want to run. 
		self.pred_op = self.graph.get_tensor_by_name("real_predict:0")
		self.top = 48.9513226
		self.right = -68.555283
		self.bottom = 28.126187
		self.left = -124.69542

	def Prepare(self, in_dict):
		vec_len = self.X.get_shape().as_list()[1]
		x = np.zeros((1, vec_len))
		for entry in in_dict.iteritems():
			if (entry[0] in self.cat_dicts.keys()):
				print(entry[0])
				ind = self.cat_dicts[entry[0]][entry[1]]
				x[0, ind] = 1;
			elif (entry[0] in self.cont_list):
				ind = self.cont_list.index(entry[0])
				x[0, ind] = entry[1]
		return x
	def Predict(self, in_dict):
		x = self.Prepare(in_dict)
		
		#print(x)
		pred = self.sess.run([self.pred_op,], feed_dict={self.X: x, self.dropout_rate: 1})
		return pred

	def HeatMap(self, in_dict, reso = 100):
		x = self.Prepare(in_dict)
		x_mat = np.ones((np.power(reso,2),1)) * x

		step = 0 
		for i in np.arange(self.top, self.bottom, (self.top - self.bottom)/reso):
			for j in np.arange(self.left, self.right, (self.right - self.left)/reso):
				x_mat[step,0] = i;
				x_mat[step,1] = j;
				step+=1

		pred = self.sess.run([self.pred_op,], feed_dict={self.X: x_mat, self.dropout_rate: 1})
		return np.reshape(pred, (reso, reso))



if __name__ == "__main__":
	model = NN_Model()
	"""tests = [
	{"lat": 36.1626638,
	"lng": -86.78160159999999,
	"supervisingJob": 1,
	'educationRequirement': 1,
	'normTitleCategory': 'food'},

	{"lat": 36.2626638,
	"lng": -86.78160159999999,
	"supervisingJob": 1,
	'educationRequirement': 1,
	'normTitleCategory': 'food'},

	{"lat": 36.0626638,
	"lng": -86.78160159999999,
	"supervisingJob": 1,
	'educationRequirement': 1,
	'normTitleCategory': 'food'},

	{"lat": 36.1626638,
	"lng": -86.88160159999999,
	"supervisingJob": 1,
	'educationRequirement': 1,
	'normTitleCategory': 'food'},

	{"lat": 36.1626638,
	"lng": -86.68160159999999,
	"supervisingJob": 1,
	'educationRequirement': 1,
	'normTitleCategory': 'food'}
	]"""

	test = {"lat": 36.1626638,
	"lng": -86.78160159999999,
	"supervisingJob": 1,
	'educationRequirement': 1,
	'normTitleCategory': 'food'}

	#for test in tests:
	prediction = model.HeatMap(test)
	print('\n\n')

	print('Estimated Salary:')
	print(prediction)
