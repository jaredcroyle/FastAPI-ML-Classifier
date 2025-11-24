import torch
import torch.nn as nn
import torch.nn.functional as F

class GenomicsTabularNN(nn.Module):
    def __init__(self, input_size=180, num_classes=4):  # use 180 features + 1 class
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)