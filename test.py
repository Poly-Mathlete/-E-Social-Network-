import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")
print("Model loaded!")