from numpy.linalg import lstsq

x = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]

print (lstsq(x, y, rcond=None)[0])