
from random import randrange

#### RANDOM FOREST IMPLEMENTATION ####

# On UCI Ionosphere Dataset http://archive.ics.uci.edu/ml/datasets/Ionosphere #

# Split dataset based on an attribute and an attribute value
def testSplit(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def giniIndex(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def getSplit(dataset, nFeatures):
	classValues = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < nFeatures:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = testSplit(index, row[index], dataset)
			gini = giniIndex(groups, classValues) 
			if gini < b_score: # Decision point
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, maxDepth, minSize, nFeatures, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= maxDepth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= minSize:
		node['left'] = to_terminal(left)
	else:
		node['left'] = getSplit(left, nFeatures)
		split(node['left'], maxDepth, minSize, nFeatures, depth+1)
	# process right child
	if len(right) <= minSize:
		node['right'] = to_terminal(right)
	else:
		node['right'] = getSplit(right, nFeatures)
		split(node['right'], maxDepth, minSize, nFeatures, depth+1)

# Build a decision tree
def buildTree(train, maxDepth, minSize, nFeatures):
	root = getSplit(train, nFeatures)
	split(root, maxDepth, minSize, nFeatures, 1)
	return root

# Make prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	nSample = round(len(dataset) * ratio)
	while len(sample) < nSample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make prediction with a list of bagged trees
def baggingPredict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Random Forest algorithm
def randomForest(train, test, maxDepth, minSize, sampleSize, nTrees, nFeatures):
	trees = list()
	for i in range(nTrees):
		sample = subsample(train, sampleSize)
		tree = buildTree(sample, maxDepth, minSize, nFeatures)
		trees.append(tree)
	predictions = [baggingPredict(trees, row) for row in test]
	return(predictions)