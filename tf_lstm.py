# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:00:52 2019

@author: rajkumar.rajasekaran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X=np.linspace(0,10,num=200)

y=np.sin(X)





plt.plot(X,y)
plt.title('Time series input')


df= pd.DataFrame({'X':X,'Y':y})

#df.set_index('X',inplace= True)

import tensorflow as tf

#lets create a helper function for batching our data
def next_batch(training_data,batch,steps):
        ran = np.random.randint(0,len(training_data)-steps)
        y_batch= np.array(training_data[ran:ran+steps+1]).reshape(1,steps+1)
        return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)

#lets initilse our variables and keepour batch size as 10 and output size as 1
X = tf.placeholder(tf.float32,[None,8,1])
y = tf.placeholder(tf.float32,[None,8,1])

#lets add our LSTM cell with a Relu activation function
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.LSTMCell(num_units = 100 , activation = tf.nn.relu),output_size = 1)

#add variables for the outsputs and states    
outputs, states = tf.nn.dynamic_rnn(cell,X,dtype = tf.float32)

#addour loss function mse error
loss = tf.reduce_mean(tf.square(outputs-y))

#add our optimizer ad Adam
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

#minimise the loss function
train = optimizer.minimize(loss)


#initilaise the global variables
init = tf.global_variables_initializer()

#invoke the session
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
            x_batch,y_batch = next_batch(df[['Y']],1,8)
            sess.run(train,feed_dict={X:x_batch,y:y_batch})
    
    
#for prediction lets take points within the limit so we can visuvalise better
    t_s =  df.Y.iloc[50:58,].values.tolist()
    for i in range(20):
            x_b=np.array(t_s[-8:]).reshape(-1,8,1)
            y_pred=sess.run(outputs,feed_dict={X:x_b})
            t_s.append(y_pred[0, -1, 0])

pred=t_s[-20:]        
    
#lets plot our output and input
plt.plot(df.X,df.Y)
plt.plot(df.X.iloc[58:78,].values.tolist(),df.Y.iloc[58:78,].values.tolist(),label='Actual')
plt.plot(df.X.iloc[58:78,].values.tolist(),pred,label='Predictions')
plt.title('LSTM- Timeseries Forecast')
plt.xlabel('time')
plt.ylabel('values')
plt.legend()

