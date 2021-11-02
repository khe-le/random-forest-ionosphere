from random import seed
from random import randrange
from math import sqrt
from randomForest import randomForest
from csv import reader

##### RANDOM FOREST VALIDATION ####

# Read data from CSV file
def readCSV(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csvReader = reader(file)
		for row in csvReader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def strColToFloat(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def strColToInt(dataset, column):
	classValues = [row[column] for row in dataset]
	unique = set(classValues)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split dataset into k folds
def crossValidationSplit(dataset, nFolds):
	datasetSplit = list()
	datasetCopy = list(dataset)
	foldSize = int(len(dataset) / nFolds)
	for i in range(nFolds):
		fold = list()
		while len(fold) < foldSize:
			index = randrange(len(datasetCopy))
			fold.append(datasetCopy.pop(index))
		datasetSplit.append(fold)
	return datasetSplit

# Calculate accuracy percentage
def accuracyMetric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate algorithm using cross validation split
def evaluate(dataset, algorithm, nFolds, *args):
	folds = crossValidationSplit(dataset, nFolds)
	scores = list()
	for fold in folds:
		trainSet = list(folds)
		trainSet.remove(fold)
		trainSet = sum(trainSet, [])
		testSet = list()
		for row in fold:
			rowCopy = list(row)
			testSet.append(rowCopy)
			rowCopy[-1] = None
		predicted = algorithm(trainSet, testSet, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracyMetric(actual, predicted)
		scores.append(accuracy)
	return scores

if __name__ == '__main__':
    # testing the random forest algorithm
    seed(2)
    filename = 'ionosphere.data.csv' # load and prepare data
    dataset = readCSV(filename)
    # convert string attributes to integers
    for i in range(0, len(dataset[0])-1):
        strColToFloat(dataset, i)
    # convert class column to integers
    strColToInt(dataset, len(dataset[0])-1)
    # evaluate algorithm
    nFolds = 5
    maxDepth = 10
    minSize = 1
    sampleSize = 1.0
    nFeatures = int(sqrt(len(dataset[0])-1)) #
    for nTrees in [1, 3, 5]:
        scores = evaluate(dataset, randomForest, nFolds, maxDepth, minSize, sampleSize, nTrees, nFeatures)
        print('Trees: %d' % nTrees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))