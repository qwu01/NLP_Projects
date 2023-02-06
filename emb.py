import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Load the model
tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
model = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

# Prepare some text to embed
text = [
    "Chemical Formula: C13H21NO3",
]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(inputs)
# Embed the text
with torch.no_grad():
    sequence_output = model(**inputs)[0]

# Mean pool the token-level embeddings to get sentence-level embeddings
embeddings = torch.sum(
    sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

# Compute a semantic similarity via the cosine distance
# print(embeddings)

for ts in inputs["input_ids"][0]:
    print(tokenizer.decode(ts))
# xx = tokenizer.decode()
# print(xx)

# [CLS]
# chemical
# formula
# :
# c
# ##13
# ##h
# ##21
# ##no
# ##3
# [SEP]





# Prepare some text to embed
text = [
    "Chemical Formula: C13H21NO3",
]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(inputs)
inputs.to('cuda')
model.to('cuda')
# Embed the text
with torch.no_grad():
    sequence_output = model(**inputs)[0]

# Mean pool the token-level embeddings to get sentence-level embeddings
embeddings = torch.sum(
    sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

# Compute a semantic similarity via the cosine distance
# print(embeddings)

for ts in inputs["input_ids"][0]:
    print(tokenizer.decode(ts))
# xx = tokenizer.decode()
# print(xx)

### [CLS]
### chemical
### formula
### :
### c
### ##13
### ##h
### ##21
### ##no
### ##3
### [SEP]
