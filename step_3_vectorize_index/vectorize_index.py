from gensim.models import Word2Vec
import numpy as np
import json
import concurrent.futures

print("Model loading...")
model = Word2Vec.load("../step_2_generate_vectors/trained.model")
word_vectors = model.wv
del model
print("Model loaded.")

def get_word_list_vector (word_list): # sentence is an array of words
  word_list = [word for word in word_list if word in word_vectors.index_to_key]
  return np.mean(word_vectors[word_list], axis=0)

print("Loading index...")
with open('../step_1_generate_corpus/index.json', 'r') as file:
  old_index = json.load(file)
print("Original index read.")

print("Loading corpus...")
with open('../step_1_generate_corpus/corpus.json', 'r') as file:
  corpus = json.load(file)
print("Generated corpus read.")

print("Splitting up corpus...")
def chunks (lst, n, length):
  for i in range(0, length, n):
    yield lst[i:i + n]
section_count = 15
corpus_length = len(corpus)
corpus_sections = chunks(list(enumerate(corpus)), int(corpus_length / section_count), corpus_length)
print("Corpus split.")
# print([len(c) for c in corpus_sections])

complete_new_index = []
def vectorize_corpus_loop (corpus_section):
  new_index = []
  section_length = float(len(corpus_section))
  indices = []
  for i, sentence in corpus_section: # remember, each array in the corpus represents a product at that spot in our index
    indices.append(i)
    if len(sentence) > 0:
      record = old_index[i]
      record["vector"] = get_word_list_vector(sentence).tolist()
      new_index.append(record)
    if (i % 500 == 0):
      current_section_index = len(new_index)
      print(str(current_section_index) + " done of " + str(int(section_length)) + ": " + str((current_section_index/section_length) * 100) + "% complete")
  complete_new_index.extend(new_index)

print("Starting threads to vectorize corpus...")
with concurrent.futures.ThreadPoolExecutor(max_workers=section_count) as executor:
  executor.map(vectorize_corpus_loop, corpus_sections)
print("Corpus vectorized.")

print("Saving new index...")
with open('vectorized_index.json', 'w') as file:
  file.write(json.dumps(complete_new_index, indent=4))
print("New index saved.")