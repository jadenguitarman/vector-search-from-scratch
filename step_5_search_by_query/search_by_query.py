from gensim.models import Word2Vec, KeyedVectors
import sys, json
from pprint import pprint
import numpy as np
import time

print("Loading initial trained model...")
sentence_vectors = Word2Vec.load("../step_2_generate_vectors/trained.model").wv
print("Trained model loaded.")

print("Loading index model...")
product_vectors = KeyedVectors.load("../step_4_generate_index_model/index.model")
print("Index model loaded.")

print("Reading the index dictionary...")
with open('../step_4_generate_index_model/index_dict.json', 'r') as file:
  index = json.load(file)
print("Index read.")

if len(sys.argv) > 1:
  start = time.time()
  
  query = sys.argv[1:]
  print("Query: " + " ".join(query))
  query = [word for word in query if word in sentence_vectors.index_to_key]
  print("Query words used: " + " ".join(query))

  query_vector = np.mean(sentence_vectors[query], axis=0)
  similar_products = product_vectors.most_similar(query_vector, topn=10)
  pprint(similar_products, indent=4)
  search_results = [index[result[0]] for result in similar_products]
  pprint(search_results[:5], indent=4)

  end = time.time()
  print("The results were returned in " + str((end - start) * 1000.0) + " ms")
else:
  print("No query specified.")