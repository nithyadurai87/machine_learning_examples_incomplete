from sklearn.metrics.pairwise import euclidean_distances
counts = [
[[0, 1, 1, 0, 0, 1, 0, 1]],
[[0, 1, 1, 1, 1, 0, 0, 0]],
[[1, 0, 0, 0, 0, 0, 1, 0]]
]
print (euclidean_distances(counts[0], counts[1]))
print (euclidean_distances(counts[0], counts[2]))
print (euclidean_distances(counts[1], counts[2]))