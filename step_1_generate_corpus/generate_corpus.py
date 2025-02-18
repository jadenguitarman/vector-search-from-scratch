import json, re, nltk

lemma = nltk.wordnet.WordNetLemmatizer()

print("Opening the JSON file...")
input_records = []
with open('../inputs/initial_dataset.json', 'r') as file:
  print("Reading the JSON file...")
  lines = file.read().split("\n")
  for line in lines:
    try:
      input_records.append(json.loads(line))
    except:
      pass
print("JSON file read.")

print("Transforming...")
output_records = []
corpus = []
for record in input_records:
  if not "name" in record or not "brand" in record or not "description" in record or not "category" in record:
    continue
  
  output_records.append(record)

  record = re.sub('[^a-zA-Z]', ' ', record["name"]) + re.sub('[^a-zA-Z]', ' ', record["brand"]) + re.sub('[^a-zA-Z]', ' ', record["description"]) + re.sub('[^a-zA-Z]', ' ', record["category"])
  record = list(filter(None, record.lower().split(" ")))
  record = list(map(lemma.lemmatize, record))
  corpus.append(record)
print("Data transformed.")

print("Writing to the JSON output...")
with open('corpus.json', 'w') as file:
  file.write(json.dumps(corpus, indent=4))
with open('index.json', 'w') as file:
  file.write(json.dumps(output_records, indent=4))
print("JSON output written.")