import numpy as np 
import pandas as pd
from sklearn import preprocessing, cross_validation
import matplotlib.pyplot as plt
import math

def naiveBayes(data, plot_data=False, plot_x_dim=0):
	print('\n\n\n\n')
	df = pd.read_csv(data)

	X = np.array(df.drop(['result'],1))
	Y = np.array(df['result'])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.1)

	X_spam = separate_data(1,y_train,X_train,58)
	X_no_spam = separate_data(0,y_train,X_train,58)

	avg_vector_spam = np.mean(X_spam.T, axis=1)
	avg_vector_no_spam = np.mean(X_no_spam.T, axis=1)

	covariate_matrix_spam = np.cov(X_spam.T)
	covariate_matrix_no_spam = np.cov(X_no_spam.T)

	probabilities = []
	results = 0
	count = 0
	for i in X_test:
		p_1 = f_x(i, avg_vector_spam,covariate_matrix_spam, 1, y_test)
		p_0 = f_x(i, avg_vector_no_spam, covariate_matrix_no_spam, 0, y_test)
		if((p_1 > p_0 and y_test[count] == 1) or (p_0 > p_1 and y_test[count] == 0)):
	 		#Correctly classified as spam
	 		results += 1
		count += 1
	 	
	accuracy = results/len(X_test)*100
	print(float(covariate_matrix_spam[0][0]))
	print('This classifier accurately predicted whether an e-mail was spam %.5s percent of the time on 10 percent of the data\n\n\n\
		'%(accuracy))

	if(plot_data == True):
		### Plots the first dimension of x as a normal distribution for the visualization ###
		spam_dist = gaussianDist(float(avg_vector_spam[plot_x_dim]), float(covariate_matrix_spam[plot_x_dim][plot_x_dim]), -1, 1, 0.005)
		no_spam_dist = gaussianDist(float(avg_vector_no_spam[plot_x_dim]), float(covariate_matrix_no_spam[plot_x_dim][plot_x_dim]), -1, 1, 0.005)
		x = [i[0] for i in spam_dist]
		y = [i[1] for i in spam_dist]
		x2 = [i[0] for i in no_spam_dist]
		y2 = [i[1] for i in no_spam_dist]

		plot(x,y,x2,y2)
def pi(y_value, y_data):
	#indicator = lambda x,y: 1 if x == y else 0 
	mail_type_count = 0
	for i in y_data:
		if(i == y_value):
			mail_type_count +=1

	return mail_type_count/len(y_data)
def f_x(x, mean, covariance, y_value, y_data):
	mean = np.asmatrix(mean)
	covariance = np.asmatrix(covariance)
	x = np.asmatrix(x).T
	k = x.shape[0]
	w = np.asmatrix(x-mean)
	Z = np.matmul(covariance.I, w)
	constants = pi(y_value, y_data)*np.linalg.det(covariance)
	gauss_x = constants*np.exp(-0.5*(np.matmul(w.T, Z)))
	return gauss_x

##################### Returns submatrix X for spam/no spam in the training data #####################
def separate_data(y_value, y_data, X_data, dimenssions_input):
	#########Compute n_y#########
	n_y = 0 
	for i in y_data:
		if(i == y_value):
			n_y += 1

	data = []
	j = 0
	for k in range(0, dimenssions_input-1):
		tmp = []
		for i in range(0, len(X_data)):
			if(y_value != y_data[i]):
				continue
			tmp.append(X_data[i][j])
		j+=1
		data.append(np.array(tmp))
		tmp.clear()
	return np.asmatrix(data).T

def gaussianDist(mean, std_dev, start, end, increments):
	constant = 1/(math.sqrt((math.pi)*2)*std_dev)
	x = start
	results = []
	for i in np.arange(start, end, increments):
		x, f_x = x,  constant*math.exp(-( ( (x-mean)**2 )/(2*std_dev**2) ) )
		results.append([x, f_x])
		x += increments
	return results

def plot(x,y,x2,y2):
	plt.scatter(x,y, label='spam', color='r', s=10, marker='o')
	plt.scatter(x2,y2, label='no_spam', color='g', s=10, marker='o')
	plt.xlabel('x')
	plt.ylabel('Probability')
	plt.title('Gaussian distribution\n')
	plt.legend()
	plt.show()


def main():
	naiveBayes('NaiveBayesEmailData.txt', plot_data=True, plot_x_dim=0)



if __name__ == '__main__':
	main()