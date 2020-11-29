
from math import sqrt

class KNN:

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
        


# Initialize knn object
knn = KNN()
# Example dataset for testing Euclidean distance function
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
