
from math import sqrt

class KNN:
    # Euclidian distance function this calcuates the Euclidian distance 
    # between teo vectos represented as rows

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
        neighbors = list()
        for i in range(numNeighbors):
            neighbors.append(distances[i][0])
        return neighbors

# Initialize knn object
knn = KNN()
# Example dataset for testing Euclidean distance function
dataset = [[2.7810836,2.550537003,0],
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
# is between the firt row and itself and should be 0.

print("Testing euclideanDistance(): ")
row0 = dataset[0]
for row in dataset:
    distance = knn.euclideanDistance(row0,row)
    print(distance)
print("\n")

#Expierement for testing getNeighbors()
# this will print the 3 most similar elements in the dataset to
# first element in order of similarity. The first element should be 
# at the top of the list as it is most similar to itself.
print("Testing getNeighbors(): ")
neighbors = knn.getNeighbors(dataset, dataset[0], 3)
for n in neighbors:
    print(n)
