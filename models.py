import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from parameters import Hiperparameters

class MIDIModel(nn.Module):
    def __init__(self):
        super(MIDIModel, self).__init__()
        self.fc1 = nn.Linear(4, Hiperparameters.hidden_size)  # 4 giriş özelliği (start, end, pitch, velocity)
        self.fc2 = nn.Linear(Hiperparameters.hidden_size, 256)
        self.fc3 = nn.Linear(256, 4)  # 4 çıkış özelliği

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='model.bin'):
        """
        Args:
            patience (int): İyileşme olmadan beklenilecek epoch sayısı.
            verbose (bool): Erken durma mesajının yazdırılması.
            delta (float): İyileşmenin önemsenebilir kabul edilmesi için minimum değişiklik.
            path (str): Modelin kaydedileceği dosya yolu.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Doğrulama kaybı azaldığında modeli kaydet."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
