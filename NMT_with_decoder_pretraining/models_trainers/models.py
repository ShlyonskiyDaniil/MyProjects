import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, voc_size, emb_size, padding_idx, hid_size):
        super().__init__()

        self.voc_size = voc_size
        self.emb_size = emb_size
        self.padding_idx = padding_idx
        self.hid_size = hid_size

        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )
        
        self.rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=hid_size // 2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, batch): # batch: [b_size, seq_len]

        emb = self.embedding(batch) # [b_size, seq_len, emb_size]

        output, _ = self.rnn(emb)
        # output: [b_size, seq_len, hid_size]
        # h: [2, b_size, hid_size // 2]

        return output

class Decoder(nn.Module):

    def __init__(self, voc_size, emb_size, padding_idx, hid_size):
        super().__init__()

        self.voc_size = voc_size
        self.emb_size = emb_size
        self.padding_idx = padding_idx
        self.hid_size = hid_size

        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )

        self.rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=hid_size,
            batch_first=True,
            bidirectional=False
        )

        self.to_logits = nn.Linear(hid_size, voc_size)
    
    def forward(self, batch, h): # batch: [b_size, seq_len]

        emb = self.embedding(batch) # [b_size, seq_len, emb_size]

        output, _ = self.rnn(emb) # [b_size, seq_len, hid_size]
        logits = self.to_logits(output) # [b_size, seq_len, voc_size]

        return logits

class FeedForward1(nn.Module):

    def __init__(self, hid_size):
        super().__init__()

        self.model = nn.Linear(hid_size, hid_size)
    
    def forward(self, batch): # [b_size, hid_size]

        return self.model(batch)

class FeedForward2(nn.Module):

    def __init__(self, hid_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hid_size, 4 * hid_size),
            nn.ReLU(),
            nn.Linear(4 * hid_size, hid_size),
        )
    
    def forward(self, batch): # [b_size, hid_size]

        return self.model(batch)

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, feed_forward=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.feed_forward = self.init_feed_forward(encoder.hid_size) if feed_forward is None else feed_forward
    
    def init_feed_forward(self, hid_size):

        linear = nn.Linear(hid_size, hid_size, bias=False)
        linear.W = torch.eye(hid_size)
        linear.W.requires_grad = False
        
        return linear
    
    def forward(self, src, trg):
        
        # src: [b_size, seq_len]
        # trg: [b_size, seq_len]

        src_output = self.encoder(src) # [b_size, seq_len, hid_size]
        last_hidden_state_idx = (src == self.encoder.padding_idx).float().argmax(dim=-1) - 1
        src_h = src_output[:, last_hidden_state_idx, :] # [b_size, hid_size]

        trg_h = self.feed_forward(src_h) # [b_size, hid_size]

        trg_output = self.decoder(trg, trg_h) # [b_size, seq_len, hid_size]

        return trg_output