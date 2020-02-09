import numpy as np
import torch
import torch.nn.functional as F
from pdb import set_trace as bp

class CNN(torch.nn.Module):

    def __init__(self, args, vocab):
        super(CNN, self).__init__()

        self.args = args

        self.embed_size = args.embed_size
        self.filter_widths = [3, 4, 5]
        self.maps = 100

        self.embedding = torch.nn.Embedding(num_embeddings=len(vocab),\
                                            embedding_dim=self.embed_size,\
                                            padding_idx=0)

        self.convs = []
        self.pools = []
        for i, each in enumerate(self.filter_widths):
            self.convs.append(torch.nn.Conv1d(in_channels=1,\
                                              out_channels=self.maps,\
                                              kernel_size=self.embed_size*each,\
                                              stride=self.embed_size))
            self.pools.append(torch.nn.MaxPool1d(kernel_size=self.args.max_sent_len\
                               - self.filter_widths[i] + 1))

        self.convs = torch.nn.ModuleList(self.convs)
        self.pools = torch.nn.ModuleList(self.pools)

        self.final = torch.nn.Linear(self.maps * len(self.filter_widths),\
                                     len(vocab.class2id_dict))

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=args.dropout)

    def load_vector(self, args, vocab):

        print("Loading Pretrained Vectors")
        vectors = np.zeros((len(vocab), args.embed_size))
        
        fp = open(args.embed_file)
        
        for line in fp:
            word, vec = line.split(' ', 1)
            word = word.lower().strip()
            vec = np.fromstring(vec, sep=' ')
            idx = vocab[word]
            if idx != 2:
                vectors[idx] = vec 

        vectors = np.asarray(vectors)
        self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        if args.fine_tune == 0:
            self.embedding.weight.requires_grad = False

    def forward(self, examples):

        embeddings = self.embedding(examples)
        embeddings = embeddings.view(embeddings.shape[0], 1, -1)
        outputs = []
        for i in range(len(self.filter_widths)):
            output = self.relu(self.convs[i](embeddings))
            output = self.pools[i](output)
            outputs.append(output.squeeze(-1))

        output = torch.cat(outputs, dim=1)
        output = self.dropout(output)
        output = self.final(output)

        return output
