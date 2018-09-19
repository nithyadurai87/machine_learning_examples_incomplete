import pandas as pd
df = pd.read_csv('./spam.csv', delimiter='\t',header=None)
print (df.head())
print ('Number of spam messages:', df[df[0] == 'spam'][0].count())
print ('Number of ham messages:', df[df[0] == 'ham'][0].count())
