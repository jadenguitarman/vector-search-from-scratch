from gensim.models import Word2Vec
import gensim.downloader as api
import json

print("Seeding model with base corpus...")
base_corpus = api.load('text8')
model = Word2Vec(base_corpus, vector_size=100, window=5, workers=4, min_count=1)
print("Model seeded.")

print("Reading the JSON corpus file...")
with open('../step_1_generate_corpus/corpus.json', 'r') as file:
  connected_lists = json.load(file)
print("Product corpus read.")

print("Model now training with inputted corpus...")
model.train(connected_lists, total_examples=len(connected_lists), epochs=10)
print("Model trained.")

model.save("trained.model")
print("Model saved.")