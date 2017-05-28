import cv2
import numpy as np
import time
from sklearn.svm import LinearSVC

def fit_classifier(X_train, y_train):
	# Use a linear SVC
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')

	return svc

def test_classifier(svc, X_test, y_test):
	score = round(svc.score(X_test, y_test))
	print('Test Accuracy of SVC = ', round(score, 4))
	prediction = svc.predict(X_test)
	count = 0
	for i in range(len(prediction)):
		if prediction[i] == y_test[i]:
			count = count + 1

	print(count, len(prediction))
	percent = count / len(prediction)

	print('My SVC predicts:', round(percent, 4), 'percent accuracy')
