
from math import sqrt
from random import seed
from random import randrange
from csv import reader

class KNN:

    ###################################################################
    ################ Dataset Loading Helper Mehtods ###################
    ###################################################################

    # Load a CSV file
    def loadCSV(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            cReader = reader(file)
            for row in cReader:
                if not row:
                    continue
                dataset.append(row)
        return dataset
    
    # Convert String column to float
    def stringColToFloat(self, dataset, col):
        for row in dataset:
            row[col] = float(row[col].strip())
    
    # Convert String column to int
    def stringColToInt(self, dataset, col):
        classes = list()
        for row in dataset:
            classes.append(row[col])
        unique = set(classes)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
            # Print the mapping of string class names to their asciocated integer
            print("[%s] => %d" % (value, i))
        for row in dataset:
            row[col] = lookup[row[col]]
        return lookup

     # Find min and max values for each column
    def datasetMinMax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            colValues = [row[i] for row in dataset]
            minVal = min(colValues)
            maxVal = max(colValues)
            minmax.append([minVal,maxVal])
        return minmax
    
    # Rescale dataset cols to the range 0-1
    def normalizeDataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    
    ###################################################################
    ################ Algorithm Evaluation Methods #####################
    ###################################################################

    # Split dataset into k folds
    def crossValidationSplit(self, dataset, n):
        datasetSplit = list()
        datasetCopy = list(dataset)
        foldSize = int(len(dataset) / n)
        for _ in range(n):
            fold = list()
            while len(fold) < foldSize:
                i = randrange(len(datasetCopy))
                fold.append(datasetCopy.pop(i))
            datasetSplit.append(fold)
        return datasetSplit
    
    # Find accuracy percentage
    def accuracyMetric(self, actual, predicted):
        numCorrect = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                numCorrect += 1
        return numCorrect / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluateAlgorithm(self, dataset, algo, nFolds, *args):
        folds = self.crossValidationSplit(dataset, nFolds)
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
            predicted = algo(trainSet, testSet, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracyMetric(actual, predicted)
            scores.append(accuracy)
        return scores

    ###################################################################
    ################### kNN Algorithm Methods #########################
    ###################################################################

    # Euclidian distance function this calcuates the Euclidian distance 
    # between two vectos represented as rows
    def euclideanDistance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance = distance + (row1[i] - row2[i])**2
        return sqrt(distance)
    
    # Locate the nearest neighbors
    def getNeighbors(self, train, testRow, numNeighbors):
        distances = list()
        # get Euc distance between each testRow and trainRow
        for trainRow in train:
            dist = self.euclideanDistance(testRow, trainRow)
            distances.append((trainRow,dist))
        # Sort by the second element of the tuple
        distances.sort(key=lambda x: x[1])
        # Build list of most similar neighbors to testRow
        neighbors = list()
        for i in range(numNeighbors):
            neighbors.append(distances[i][0])
        return neighbors
    
    # Make a classification prediction with the nearest neighbors
    def predictClassification(self, train, testRow, numNeighbors):
        neighbors = self.getNeighbors(train, testRow, numNeighbors)
        # values = [row[-1] for row in neighbors]
        values = list()
        for row in neighbors:
            values.append(row[-1])
        prediction = max(set(values), key=values.count)
        return prediction
    
    # kNN Algorithm
    def kNearestNeighbors(self, train, test, numNeighbors):
        predictions = list()
        for row in test:
            value = self.predictClassification(train,row,numNeighbors)
            predictions.append(value)
        return predictions

###################################################################
######################## Experiments ##############################
###################################################################

# Initialize knn object
knn = KNN()

# Example datasets for testing 
dataset = [[ 0.54857876,  5.25697175,0],
       [ 7.81764176,  9.17441769,1],
       [ 3.74230831,  2.06165119,0],
       [ 8.39840247,  9.46184753,1],
       [ 4.16672282,  4.14349935,1],
       [ 5.16342338,  3.32907878,1],
       [ 0.27843358,  7.39775071,0],
       [ 1.8701541 ,  4.98218665,0],
       [ 2.23288871,  7.38405505,1],
       [ 8.94828588,  7.82797826,1]]

dataset2 = [[2.7810836,2.550537003,0],
	    [1.465489372,2.362125076,0],
	    [3.396561688,4.400293529,0],
	    [1.38807019,1.850220317,0],
	    [3.06407232,3.005305973,0],
	    [7.627531214,2.759262235,1],
	    [5.332441248,2.088626775,1],
	    [6.922596716,1.77106367,1],
	    [8.675418651,-0.242068655,1],
	    [7.673756466,3.508563011,1]]

# Expierement for testing the euclideanDistance()
# This prints the distance between the first row in the dataset 
# and all other rows. The first Euclidean distance calculated
# is between the first row and itself and should be 0.

print("Testing euclideanDistance(): ")
row0 = dataset[0]
for row in dataset:
    distance = knn.euclideanDistance(row0,row)
    print(distance)
print("\n")

# Expierement for testing getNeighbors()
# this will print the 3 most similar elements in the dataset to
# first element in order of similarity. The first element should be 
# at the top of the list as it is most similar to itself.
print("Testing getNeighbors(): ")
neighbors = knn.getNeighbors(dataset, dataset[0], 3)
for n in neighbors:
    print(n)
print("\n")

# Expierement for testing predictClassification()
# this will print the expected classification of 0 and 
# the actual classification predicted from the 3 most similar
# neighbors in the dataset.
print("Testing predictClassification(): ")
prediction = knn.predictClassification(dataset, dataset[0], 3)
print("Expected Classification: %d \nActual Classification: %d \n" % (dataset[0][-1], prediction))


# Testing the kNN algo on the Iris Flowers dataset.
print("Testing kNN Algorithm on the Iris Flowers dataset: ")

seed(1)
filename = "iris.csv"
dataset = knn.loadCSV(filename)
for i in range(len(dataset[0])-1):
    knn.stringColToFloat(dataset,i)
# Convert class colummn to integers
knn.stringColToInt(dataset, len(dataset[0])-1)
# Evaluate algorithm
nFolds = 5
numNeighbors = 5

# This expierement evaluates the kNN algorithm using k-fold cross-validation with 5 folds.
# The outputs from this expierement are the mean classification accuracy scores on each 
# cross-validation fold and the mean accuracy score.
print()
print("Testing Mean Classification Accuracy Scores and Mean Accuracy Score(): ")
scores = knn.evaluateAlgorithm(dataset, knn.kNearestNeighbors, nFolds, numNeighbors)
print("Scores: %s" % scores)
print("Mean Accuracy: %.3f%%\n" % (sum(scores)/float(len(scores))))

# Making a prediction with kNN algo on Iris Dataset.
# A new row of the dataset is defined and using the 
# predictClassification() method we can predict what
# class the new row should be defined as. 
print("Making a prediction using kNN: ")
# Define a new row
newRow = [5.7,2.9,4.2,1.3]
# Make class prediction
predictedClass =  knn.predictClassification(dataset, newRow, numNeighbors)
print("Data: %s \nPredicted: %s" % (newRow, predictedClass))

