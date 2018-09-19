from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

x = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(x,y)

x1 = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y1 = [[11], [8.5], [15], [18], [11]]

predictions = model.predict(x1)
for i, prediction in enumerate(predictions):
	print ((prediction, y1[i]))
	
plt.figure()
plt.title('Pizzapriceplottedagainstdiameter')
plt.xlabel('Diameterininches')
plt.ylabel('Priceindollars')
plt.plot(x,y,'.')
plt.plot(x,model.predict(x),'--')
plt.axis([0,25,0,25])
plt.grid(True)
plt.show()
	
print (model.score(x1, y1))

