from algomattis import predict_sentiment

print(predict_sentiment("je suis heureux "))  

import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")
print("Model loaded!")