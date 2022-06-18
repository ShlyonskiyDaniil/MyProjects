import tqdm
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output


class Seq2SeqTrainer:

    def __init__(self, seq2seq, scheduler, criterion, device, acc_steps=1):

        self.seq2seq = seq2seq.to(device)
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.acc_steps = acc_steps

        self.train_loss_history = []
        self.val_loss_history = []
    
    def train(self, train_loader, n_epochs, val_loader=None):

        for _ in range(n_epochs):

            self.train_loop(train_loader)

            if val_loader is not None:
                self.val_loader(val_loader)
    
    def train_loop(self, loader):

        self.seq2seq.train()

        acc_iter = 1

        for i, (src, trg) in tqdm.tqdm(enumerate(loader)):

            self.scheduler.optimizer.zero_grad()

            src, trg = src.to(self.device), trg.to(self.device)

            output = self.seq2seq(src, trg)
            loss = self.criterion(output[:, :-1, :].permute(0, 2, 1), trg[:, 1:])
            
            loss.backward()
            
            if acc_iter == self.acc_steps:
                self.scheduler.optimizer.step()
                acc_iter = 0
            
            acc_iter += 1

            self.train_loss_history.append(loss.item() / src.shape[0])

            if (i + 1) % 10 == 0:
                self.plot_loss()
        
    def val_loop(self, loader):

        self.seq2seq.eval()

        with torch.no_grad():

            for i, (src, trg) in enumerate(loader):

                src, trg = src.to(self.device), trg.to(self.device)

                output = self.seq2seq(src, trg)
                loss = self.criterion(output[:, :-1, :].permute(0, 2, 1), trg[:, 1:])

                self.val_loss_history.append(loss.item() / src.shape[0])

                if (i + 1) % 10 == 0:
                    self.plot_loss()

    def plot_loss(self):

        clear_output(wait=True)
        plt.plot(self.train_loss_history)
        plt.plot(self.val_loss_history)
        plt.show()


class FFTrainer:

    def __init__(self, src_encoder, trg_encoder, ff, opt, criterion, device, acc_steps):

        self.src_encoder = src_encoder.to(device)
        self.trg_encoder = trg_encoder.to(device)
        self.ff = ff.to(device)

        self.opt = opt
        self.criterion = criterion

        self.device = device
        self.acc_steps = acc_steps

        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, train_loader, n_epochs, val_loader=None):

        for _ in range(n_epochs):

            self.train_loop(train_loader)
            
            if val_loader is not None:
                self.val_loop(val_loader)
            
    def train_loop(self, loader):

        self.src_encoder.eval()
        self.trg_encoder.eval()
        self.ff.train()

        acc_iter = 1

        for i, (src, trg) in enumerate(loader):

            self.opt.zero_grad()

            src, trg = src.to(self.device), trg.to(self.device)

            with torch.no_grad():

                src_output = self.src_encoder(src) # [b_size, seq_len, hid_size]
                last_hidden_state_idx = (src == self.src_encoder.padding_idx).float().argmax(dim=-1) - 1
                src_h = src_output[:, last_hidden_state_idx, :] # [b_size, hid_size]

                trg_output = self.trg_encoder(trg) # [b_size, seq_len, hid_size]
                last_hidden_state_idx = (trg == self.trg_encoder.padding_idx).float().argmax(dim=-1) - 1
                trg_h = trg_output[:, last_hidden_state_idx, :] # [b_size, hid_size]

            pred_trg_h = self.ff(src_h)
            loss = self.criterion(pred_trg_h, trg_h)

            loss.backward()
            
            if acc_iter == self.acc_steps:
                self.opt.step()
                acc_iter = 0
            
            acc_iter += 1

            self.train_loss_history.append(loss.item() / src.shape[0])

            if (i + 1) % 10 == 0:
                self.plot_loss()

    def val_loop(self, loader):

        self.src_encoder.eval()
        self.trg_encoder.eval()
        self.ff.eval()

        with torch.no_grad():

            for i, (src, trg) in enumerate(loader):

                src, trg = src.to(self.device), trg.to(self.device)

                src_output = self.src_encoder(src) # [b_size, seq_len, hid_size]
                last_hidden_state_idx = (src == self.src_encoder.padding_idx).float().argmax(dim=-1) - 1
                src_h = src_output[:, last_hidden_state_idx, :] # [b_size, hid_size]

                trg_output = self.trg_encoder(trg) # [b_size, seq_len, hid_size]
                last_hidden_state_idx = (trg == self.trg_encoder.padding_idx).float().argmax(dim=-1) - 1
                trg_h = trg_output[:, last_hidden_state_idx, :] # [b_size, hid_size]

                pred_trg_h = self.ff(src_h)
                loss = self.criterion(pred_trg_h, trg_h)

                self.val_loss_history.append(loss.item() / src.shape[0])

                if (i + 1) % 10 == 0:
                    self.plot_loss()

    def plot_loss(self):

        clear_output(wait=True)
        plt.plot(self.train_loss_history)
        plt.plot(self.val_loss_history)
        plt.show()

