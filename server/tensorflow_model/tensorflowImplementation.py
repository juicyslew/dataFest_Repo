import tensorflow as tf
import pandas as pd
import numpy as np
import pickle as pk

#NEED TO NORMALIZE THE INPUTS

#TODO
#1 Make Helper Function to generate heatmap value in matrix across US **
#2 Get normTitle Dictionary (Done)
#3 Create Object based predictor **

# SMALLER TODO
#1 normalize inputs (Done)
#2 convert categories into one hot encoding (Done)
#3 unnormalize for predicing

# Parameters
learning_rate = 0.000001
training_epochs = 5000
batch_size = 200
save_step = 500
summy_write_step = 25
display_step = 1;
dropout_rate = .75;
lam = 0

df = pd.read_csv("../../Data/final_latlng.csv", sep=',')

colnum = 0;

#Clean continuous predictors
#print(df)
continuous_predictors = ['lat','lng', 'supervisingJob']
df_cont = df[continuous_predictors]
colnum += len(df_cont.columns)

#Clean categorical predictors
categorical_predictors = ['educationRequirement', 'normTitleCategory', 'stateProvince']
df_cat = df[categorical_predictors]
df_list = []
cat_dicts = {}
for col in df_cat:
	tmp_dict = {}
	new_df = pd.get_dummies(df_cat[col])
	for new_col in new_df:
		tmp_dict[new_col] = colnum;
		print ("%s: %i" %(new_col, colnum))
		colnum+=1
	df_list.append(new_df)
	cat_dicts[col] = tmp_dict


with open('continuous_list.pk', 'wb') as f:
	pk.dump(continuous_predictors, f);

with open('cat_dict.pk', 'wb') as f:
	pk.dump(cat_dicts, f)

#with open('cat_start_index.pk', 'wb') as f:
#	pk.dump(len(continuous_predictors), f)

df_cat = pd.concat(df_list, axis = 1)

#Clean answers
predicted = ['estimatedSalary',]
df_ans = df[predicted]
ANSMEANS = df_ans.mean()
ANSSTANDS = df_ans.std()

df_ans = (df_ans-ANSMEANS)/ANSSTANDS
print(df_ans)

df = pd.concat([df_cont, df_cat, df_ans], axis = 1)
#print(df.iloc[:, -1])

# Network Parameters
n_hidden_1 = 50 # 1st layer number of neurons
n_hidden_2 = 50 # 2nd layer number of neurons
n_output = len(df_ans.columns)
n_input = len(df.columns) - n_output # MNIST data input (img shape: 28*28)

# Start Session
sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))

# tf Graph input
X_pre = tf.placeholder(tf.float32, [None, n_input], name='X')
y = tf.placeholder(tf.float32, [None, n_output], name = 'y')
dropout_rate = tf.Variable(.75, name = 'dropout_rate', dtype = tf.float32)
ansmeans = tf.cast(tf.constant(ANSMEANS), dtype = tf.float32, name = 'ansmeans')
ansstands = tf.cast(tf.constant(ANSSTANDS), dtype = tf.float32, name = 'ansstands')
# Store layers weight & bias
#if (not convolutional):

std_dev_av = .1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev = std_dev_av), name = 'w1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev = std_dev_av), name = 'w2'),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_output], stddev = std_dev_av), name = 'w3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev = std_dev_av), name = 'b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev = std_dev_av), name = 'b2'),
    'out': tf.Variable(tf.random_normal([n_output], stddev = std_dev_av), name = 'b3')
}



def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), dropout_rate)

    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), dropout_rate)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



def variable_summaries(var):
  #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

f = multilayer_perceptron(X_pre)
real_predict = tf.add(tf.multiply(f, ansstands), ansmeans, name='real_predict')
#tf.summary.scalar("estimatedSalary", f);

regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out'])
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.squared_difference(f, y) +  lam * regularizer, name = 'loss_Reg') #Scale this by # of weights
loss_noReg = tf.reduce_mean(tf.squared_difference(f, y), name = 'loss_noReg') #Scale this by # of weights
tf.summary.scalar("loss", loss_op);
#variable_summaries(loss_op)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

merged = tf.summary.merge_all()
sum_writer = tf.summary.FileWriter("logs", sess.graph)
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

sess.run(init)
saver.save(sess, 'mlModel/model')
saver.save(sess, 'mlModel/model', global_step=save_step, write_meta_graph=False)





step = 0
# Training cycle
for epoch in range(training_epochs):
	simple_df = df.sample(frac = 1)
	avg_cost = 0.
	total_batch = int(simple_df.shape[0]/batch_size)
	# Loop over all batches
	for i in range(total_batch):
		#batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = df.iloc[i*batch_size:(i+1)*batch_size, :n_input].values
		batch_y = df.iloc[i*batch_size:(i+1)*batch_size, n_input:].values
		#print(batch_y)
		#print(batch_x)
		#print(batch_y)


		# Run optimization op (backprop) and cost op (to get loss value)
		"""if (step % encoding_print_step == 0):
			dec_summy = sess.run(dec_summary, feed_dict={enc3.outputs: np.ones((1,encoding_size))}) # Y is same as X
			sum_writer.add_summary(dec_summy, step);"""
		r_pred = sess.run([real_predict], feed_dict={X_pre: batch_x});
        
		if (step % summy_write_step == 0):
			_, c, summy = sess.run([train_op, loss_op, merged], feed_dict={X_pre: batch_x, y: batch_y}) # Y is same as X
			sum_writer.add_summary(summy, step)
		else:
			_, c = sess.run([train_op, loss_op], feed_dict={X_pre: batch_x, y: batch_y}) # Y is same as X
		# Compute average loss
		avg_cost += c / total_batch

		step += 1;
    # Display logs per epoch step
	if epoch % display_step == 0:
		print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))


saver.save(sess, 'mlModel/model')
print("Optimization Finished!")

# Test model
inputs = df.iloc[:, :n_input].values
actual = df.iloc[:, n_input:].values

#pred =  # Apply softmax to logits
#tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
finCost = sess.run([loss_noReg], feed_dict={X_pre: inputs, y: actual})

print(finCost)
# Calculate accuracy
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
