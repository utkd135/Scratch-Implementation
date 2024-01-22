import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np

# this function returns sinusoidal embedding
def get_sinusoidal_embedding(position, d_model):
    """
    position: Position/index of the word in the text.
    d_model: input vector dimension.
    """
    embedding = np.zeros(d_model)
    for i in range(d_model):
        if i % 2 == 0:  # even indices
            embedding[i] = np.sin(position / (10000 ** (i / d_model)))  # assume i=2i from the equations
        else:  # odd indices
            embedding[i] = np.cos(position / (10000 ** ((i - 1) / d_model)))
    return torch.tensor(embedding)

# here a complete embedding is constructed which is then added to the tokens
def generate_positional_embeddings(seq_length, d_model):
    """
    seq_length: number of words in the text.
    d_model: input vector dimension.
    """
    embeddings = torch.zeros((seq_length, d_model))
    for pos in range(seq_length):
        embeddings[pos] = get_sinusoidal_embedding(pos, d_model)
    return embeddings


# MHA - Multi head attention mechanism
class Transformer(nn.Module):
    def __init__(self, d_model, head, device):
        super(Transformer, self).__init__()
        # dimension 
        self.d_model = d_model 
        # number of heads into which embedding is broken into
        self.head = head
        self.device = device
        self.Ln1 = nn.LayerNorm(d_model)

        self.QueryMat = nn.ModuleList([nn.Linear(self.d_model//self.head, self.d_model//self.head) for _ in range(self.head)])
        self.ValueMat = nn.ModuleList([nn.Linear(self.d_model//self.head, self.d_model//self.head) for _ in range(self.head)])
        self.KeyMat = nn.ModuleList([nn.Linear(self.d_model//self.head, self.d_model//self.head) for _ in range(self.head)])

        self.Ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batches):
        A = torch.zeros((batches.shape[0], 50, self.d_model)).to(self.device)
        normed_batches = self.Ln1(batches)
        for idx, batch in enumerate(normed_batches):
            for head in range(self.head):
                batch_divided = batch[:, (self.d_model//self.head)*head:(self.d_model//self.head)*(1+head)]
                W_q = self.QueryMat[head]
                W_k = self.KeyMat[head]
                W_v = self.ValueMat[head]
                q, k, v = W_q(batch_divided), W_k(batch_divided), W_v(batch_divided)
                A[idx, :, (self.d_model//self.head)*head:(self.d_model//self.head)*(1+head)] = (self.softmax((q @ k.T)/self.d_model**(-0.5)) @ v)
        batches = torch.add(batches, A)
        normed_batches = self.Ln2(batches)
        normed_batches = self.mlp(normed_batches)
        batches = torch.add(batches, normed_batches)
        return batches


class ViT(nn.Module):
    def __init__(self, in_c, N, batch_size, seq_len=49, device='cuda', d_model=16, head=2, out_channels=10):
        super(ViT, self).__init__()
        self.in_c = in_c
        self.head = head
        self.N = N
        self.device = device
        self.seq_len = seq_len
        self.d_model = d_model
        # learnable class token
        # this token is used at the end to come up with classification scores and all other tokens are discarded. 
        self.class_token = nn.Parameter(torch.randn(1, self.d_model)) 
        self.batch_size = batch_size
        self.rearrange1 = Rearrange('bs a p1 p2 s1 s2 -> bs (a p1 p2) (s1 s2)')
        self.rearrange2 = Rearrange('bs (p1 p2) s -> bs (p1 p2) s', bs=self.batch_size, p1=7, p2=7)
        self.linear_proj = nn.Linear(4 * 4, d_model, False)
        self.transformer = Transformer(self.d_model, self.head, self.device)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, out_channels),
            nn.Softmax(dim=-1)
        )

    def patchify(self, x):
        print(x.unfold(2, size=4, step=4).shape)
        x = x.unfold(2, size=4, step=4).unfold(3, size=4, step=4)  # ([10, 1, 7, 7, 4, 4])
        return x

    def add_positional_embedding(self, x):
        pos_embedding = generate_positional_embeddings(self.seq_len+1, self.d_model)
        pos_embedding = pos_embedding.repeat(x.shape[0], 1).reshape(x.shape[0], x.shape[1], x.shape[2])
        x += pos_embedding.to(self.device)
        return x

    def forward(self, x):
        patched = self.patchify(x)
        linear_patched = self.rearrange1(patched)
        linear_proj = self.linear_proj(linear_patched)
        linear_proj = self.rearrange2(linear_proj)
        linear_proj = torch.stack([torch.vstack((self.class_token, linear_proj[i])) for i in range(linear_proj.shape[0])])
        linear_proj = self.add_positional_embedding(linear_proj)
        out = self.transformer(linear_proj)
        out = out[:, 0]
        return self.mlp(out)


if __name__ == "__main__":
    vit = ViT(1, 10, 2)
    x = torch.randn(2, 1, 28, 28)
    y = vit(x)
    print(y.shape)
