#import nltk
#nltk.download()
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print (stemmer.stem('gathering'))