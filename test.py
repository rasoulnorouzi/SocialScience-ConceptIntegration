# %%
import torch
from transformers import AutoTokenizer
# %%
# add vanila bert by huggingface
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
concept = "generalization"
inputs = tokenizer(concept, return_tensors="pt")
print(inputs)   
# %%
# show the tokens
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# %%
# get emebeddings from the model of the first token of concept from last hidden layer
from transformers import AutoModel
model = AutoModel.from_pretrained(model_name)
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
# %%
