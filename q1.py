import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
from pprint import pprint
import string
from sklearn.manifold import TSNE
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open the file and read the content
with open('shakespeare_input.txt', 'r') as file:
    text = file.read()

# Convert the text to lower case
text = text.lower()

# Create a translation table that maps every punctuation character to None
translator = str.maketrans('', '', string.punctuation)

# Remove punctuation from the text
text = text.translate(translator)

# Split the text into words
words = text.split()

# Remove words having non alphabets
words = [word for word in words if word.isalpha()]

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


block_size = 5 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in (words[:]):
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    # print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append

# Move data to GPU

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

emb_dim = 4
emb = torch.nn.Embedding(len(stoi), emb_dim)

def plot_emb(emb, itos):
    fig = plt.figure(figsize=(10, 10))

    if emb.weight.shape[1] == 2:
        for i in range(len(itos)):
            x, y = emb.weight[i].detach().cpu().numpy()
            plt.scatter(x, y, color='k')
            plt.text(x + 0.5, y + 0.5, itos[i])
        plt.title('Embedding visualization')
    else: 
        tsne = TSNE(n_components=2, perplexity=24, random_state=0)
        X_tsne = tsne.fit_transform(emb.weight.detach().numpy())

        for i in range(len(itos)):
            x, y = X_tsne[i, 0], X_tsne[i, 1]
            plt.scatter(x, y, color='k')
            plt.text(x + 0.5, y + 0.5, itos[i])
        plt.title('t-SNE visualization')
        plt.xlabel('First t-SNE')
        plt.ylabel('Second t-SNE')
    
    return fig

class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x
# Generate names from untrained model


model = NextChar(block_size, len(stoi), emb_dim, 10).to(device)
# model = torch.compile(model)

g = torch.Generator()
g.manual_seed(4000002)
def predit_k(model, itos, stoi, block_size, input_str, k):
    # context = torch.zeros(block_size, dtype=torch.int, requires_grad=False).to(device)
    context = [0]*block_size
    sub_str = input_str[max(-block_size, -len(input_str)):]
    context[max(-block_size, -len(input_str)):] = [stoi[ch] for ch in sub_str]
    # for i in range(max(-block_size, -len(input_str)), 0, 1):
    #   ch = input_str[i]
    #   ix = stoi[ch]
    #   context[i] = ix
    #   print(context[i])
    output = ''

    for i in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '.':
            # break
            output += ' '
        else:
          output += ch
        # context = torch.cat((context[1:], torch.tensor([ix], dtype=torch.long).to(device)))
        context = context[1:] + [ix]
    return output


loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
import time
# Mini-batch training
batch_size = 4096
elapsed_time = []
for epoch in range(1000):
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i+batch_size]
        y = Y[i:i+batch_size]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

k = 3
k = st.number_input("Enter a whole number", value=0, step=1, placeholder="Type a number...")
st.write('The chosen k(number of next characters) is ', k)

text_input = st.text_input(
        "Enter some text to generate the next characters:",
        placeholder=st.session_state.placeholder,
    )

if text_input:
    st.session_state.placeholder = text_input
    st.write('The next ', k, ' chars are:', predit_k(model, itos, stoi, block_size, text_input, k))

st.subheader('Embedding visualization')
st.pyplot(plot_emb(emb, itos))
