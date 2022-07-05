import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR


class Seq2Seq(pl.LightningModule):

    def __init__(self, encoder, decoder, feed_forward=None, criterion=None, optimizer_parameters=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.feed_forward = self.init_feed_forward(encoder.hid_size) if feed_forward is None else feed_forward

        self.criterion = criterion
        self.optimizer_parameters = optimizer_parameters

        self.model_current_epoch = 0
    
    def init_feed_forward(self, hid_size):

        linear = nn.Linear(hid_size, hid_size, bias=False)
        linear.weight.item = torch.eye(hid_size)
        linear.weight.requires_grad = False

        return linear.eval()
    
    def forward(self, src, trg):
        
        # src: [b_size, seq_len]
        # trg: [b_size, seq_len]

        src_output = self.encoder(src) # [b_size, seq_len, hid_size]
        last_hidden_state_idx = (src == self.encoder.padding_idx).float().argmax(dim=-1) - 1
        src_h = src_output[:, last_hidden_state_idx, :] # [b_size, hid_size]

        trg_h = self.feed_forward(src_h) # [b_size, hid_size]

        trg_output = self.decoder(trg, trg_h) # [b_size, seq_len, hid_size]

        return trg_output
    
    def _run_step(self, batch, batch_idx):

        src, trg = batch

        output = self(src, trg)
        loss = self.criterion(output[:, :-1, :].permute(0, 2, 1), trg[:, 1:])

        return loss

    def training_step(self, batch, batch_idx):

        return self._run_step(batch, batch_idx)
    
    def training_epoch_end(self, outputs):

        avg_train_loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))

        self.logger.experiment.add_scalar('epoch_loss/training', avg_train_loss, self.model_current_epoch)
        self.model_current_epoch += 1
    
    def validation_step(self, batch, batch_idx):
    
        return self._run_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        avg_val_loss = torch.mean(torch.tensor([output for output in outputs]))

        self.log('val_loss', avg_val_loss)
        self.logger.experiment.add_scalar('epoch_loss/validation', avg_val_loss, self.model_current_epoch)
    
    def configure_optimizers(self):

        opt = torch.optim.Adam(self.parameters(), lr=self.optimizer_parameters['start_lr'])

        if self.optimizer_parameters.get('start_factor') is None:

            return opt
        
        else:
            lr_scheduler = LinearLR(
                opt,
                start_factor=self.optimizer_parameters['start_factor'],
                total_iters=self.optimizer_parameters['total_iters']
            )

            return [opt], [lr_scheduler]

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
        
class FeedForward(pl.LightningModule):

    def __init__(self, src_encoder, trg_encoder, criterion, optimizer_parameters):
        super().__init__()

        self.src_encoder = src_encoder
        self.trg_encoder = trg_encoder
        self.criterion = criterion
        self.optimizer_parameters = optimizer_parameters

        self.model = self._init_model()

        self.model_current_epoch = 0
    
    def _init_model(self):
        pass
    
    def forward(self, batch): # [b_size, hid_size]

        return self.model(batch)
    
    def _get_hidden_state(self, batch, padding_idx): # [b_size, seq_len]

        with torch.no_grad():
            
            output = self.src_encoder(batch) # [b_size, seq_len, hid_size]
                
            dim0, dim1, dim2 = output.shape
            last_hidden_state_idx = (batch == padding_idx).float().argmax(dim=-1) - 1
            idcs = last_hidden_state_idx.reshape(dim0, 1).expand(dim0, dim2)[:, None, :] 

            h = torch.gather(output, dim=1, index=idcs).squeeze() # [b_size, hid_size]

            return h

    def training_step(self, batch, batch_idx):

        src, trg = batch

        src_h = self._get_hidden_state(src, self.src_encoder.padding_idx)
        trg_h = self._get_hidden_state(trg, self.trg_encoder.padding_idx)

        pred_trg_h = self(src_h)
        loss = self.criterion(pred_trg_h, trg_h)

        return loss
    
    def training_epoch_end(self, outputs):

        avg_train_loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))

        self.logger.experiment.add_scalar('epoch_loss/training', avg_train_loss, self.model_current_epoch)
        self.model_current_epoch += 1
    
    def validation_step(self, batch, batch_idx):

        src, trg = batch

        src_h = self._get_hidden_state(src, self.src_encoder.padding_idx)
        trg_h = self._get_hidden_state(trg, self.trg_encoder.padding_idx)

        with torch.no_grad():
            pred_trg_h = self(src_h)

        loss = self.criterion(pred_trg_h, trg_h)

        return loss

    def validation_epoch_end(self, outputs):

        avg_val_loss = torch.mean(torch.tensor([output for output in outputs]))

        self.log('val_loss', avg_val_loss)
        self.logger.experiment.add_scalar('epoch_loss/validation', avg_val_loss, self.model_current_epoch)
    
    def configure_optimizers(self):

        opt = torch.optim.Adam(self.parameters(), **self.optimizer_parameters)

class FeedForward1(FeedForward):

    def __init__(self, src_encoder, trg_encoder, criterion, optimizer_parameters):
        super().__init__(src_encoder, trg_encoder, criterion, optimizer_parameters)
    
    def _init_model(self):

        in_size = self.src_encoder.hid_size
        out_size = self.trg_encoder.hid_size

        self.block1 = self.block(in_size, in_size)
        self.block2 = self.block(in_size, out_size)
    
    def block(self, in_size, out_size):

        return nn.Sequential(
            nn.Linear(in_size, 4 * in_size),
            nn.ReLU(),
            nn.BatchNorm1d(4 * in_size),
            nn.Linear(4 * in_size, out_size)
        )
    
    def forward(self, batch):

        x = self.block1(batch)
        x = self.block2(x + batch)

        return x
