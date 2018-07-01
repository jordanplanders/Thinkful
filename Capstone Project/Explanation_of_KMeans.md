
## Supervised v. unsupervised learning
Supervised learning algorithms require a training set of both the data you might enter and the output you would expect the final model to return. Probably the most familiar case is using a set of (x,y) points to train a linear regression model. Once the data and outputs have been churned through the algorithm, you can feed it another point with an unknown output value and the model will return its best guess. 

Unsupervised learning algorithms sift structure out of data sets for which there is no output information.  

## Clustering
Clustering algorithms take in data data and group the points into some number of clusters and return labels corresponding to those clusters.  One might make the argument that clustering is nominally supervised since many algorithms require the user to specify a number of clusters a priori, but it is also possible to train up multiple models with the same data and different numbers of specified clusters and pick the one with the best statistics...potentially eliminating the need for a best guess about the number of clusters that should be present.  

There are a number of different algorithms for clustering, but K-means clustering is relatively straight forward to explain and so we'll start there. 

### Pseudocode
It goes like this (although the actual code inevitably is written to be more efficient and less straightforward to read):

1. Randomly generate k points (for k clusters)

2. For all points: calculate the distance from point to each of the clusters and assign the point to the cluster which is the minimum distance away
3. Revise the location of each cluster by averaging the x and y values of all the points associated with the cluster

Repeat until the locations of the clusters stabilize. 


Figure | Caption
------------ | -------------
![Initial clustering output](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/cluster_figs/cluster_ex_init.png) | Left is a figure showing two points randomly generated as initial centers (the red and blue xs) and the initial labels assigned to each of the data points.
![Final clustering output](https://github.com/jordanplanders/Thinkful/blob/master/Capstone%20Project/cluster_figs/cluster_ex_final.png)| Left is a figure showing the final location of the centers with the final labels assigned to each of the data points.  You'll notice that a number of the data points have been relabelled in the process.

