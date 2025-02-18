from gensim.models import KeyedVectors
import json
import numpy as np

print("Loading index...")
with open('../step_3_vectorize_index/vectorized_index.json', 'r') as file:
  index_array = json.load(file)
print("Vectorized index read.")

print("Processing index...")
objectIDs = []
vectors = []
index = {}
for record in index_array:
  objectID = record["objectID"]
  vector = record["vector"]
  del record["objectID"]
  del record["vector"]
  index[objectID] = record
  objectIDs.append(objectID)
  vectors.append(np.array(vector))
print("Index processed.")

print("Generating model...")
vectors = np.array(vectors)
model = KeyedVectors(vectors.shape[1])
model.add_vectors(objectIDs, vectors)
print("Model generated.")

model.save("index.model")
print("Model saved.")

print("Saving new index...")
with open('index_dict.json', 'w') as file:
  file.write(json.dumps(index, indent=4))
print("New index saved.")