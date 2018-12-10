# We only care about columns 1 and 3 (Sentiment, and SentimentText)

import numpy as np

data = np.genfromtxt("SentimentAnalysisDataset.csv", delimiter=',', skip_header=1, usecols=(1,3), max_rows=(1000), dtype=None)

print(data)
#print(data[0])
print(data.shape)
