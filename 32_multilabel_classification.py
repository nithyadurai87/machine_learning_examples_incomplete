import numpy as np
from sklearn.metrics import hamming_loss,jaccard_similarity_score
print (hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[0.0, 1.0], [1.0, 1.0]])))
print (hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [1.0, 1.0]])))
print (hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [0.0, 1.0]])))
print (jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[0.0, 1.0], [1.0, 1.0]])))
print (jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [1.0, 1.0]])))
print (jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),np.array([[1.0, 1.0], [0.0, 1.0]])))
