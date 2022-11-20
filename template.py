#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/Mindarlynn/OSS-Project2

import sys
import pandas as pd

# For Accuracy, Prescision and Recall score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_dataset(dataset_path):
	#To-Do: Implement this function
	return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	sz = dataset_df.groupby("target").size()
	# dataset의 column중 마지막 한 개는 target이므로 column size - 1이 feature의 갯수
	return dataset_df.shape[1] - 1, sz[0], sz[1]

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x = data_df.drop(columns="target", axis=1)
	y = data_df["target"]
	return train_test_split(x, y ,test_size = testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	return train_test_with_model(
		DecisionTreeClassifier(),
		x_train, x_test, y_train, y_test
	)

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	return train_test_with_model(
		RandomForestClassifier(),
		x_train, x_test, y_train, y_test
	)

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)
	return train_test_with_model(
		pipe,
		x_train, x_test, y_train, y_test
	)

def train_test_with_model(model, x_train, x_test, y_train, y_test):
	model.fit(x_train, y_train)
	predicted = model.predict(x_test)

	return \
		accuracy_score(y_test, predicted), \
		precision_score(y_test, predicted), \
		recall_score(y_test, predicted)

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)