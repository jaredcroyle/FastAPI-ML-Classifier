import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from app.model import GenomicsTabularNN
from app.encoding import load_csv_data

config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'num_workers': 4,
    'validation_split': 0.2,
    'random_seed': 42,
    'model_save_path': 'models/splice_junction_classifier.pth',
    'num_classes': 4  # update this based on your dataset!!
}

def load_and_prepare_data(csv_path='dna.csv'):
    """Load and prepare the dataset."""
    # loading dataset
    df = pd.read_csv(csv_path)
    
    # separating features and labels
    X = df.drop('class', axis=1).values.astype(np.float32)
    y = df['class'].values.astype(np.int64)
    
    # converts to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    # creating dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # splits into train and validation sets
    val_size = int(len(dataset) * config['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )
    
    # creates data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    return train_loader, val_loader, X.shape[1]

def train_model():
    # setting up random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    
    # loads data
    train_loader, val_loader, input_size = load_and_prepare_data()
    
    # initializes model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GenomicsTabularNN(input_size=input_size, num_classes=config['num_classes'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        # training in progress
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zeroes the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # stats for training
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # validation
        val_accuracy, val_loss = validate(model, val_loader, criterion, device)
        
        # print statistics
        print(f'Epoch: {epoch+1:03d}/{config["num_epochs"]} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
        
        # saves the best model
        if val_accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:.2f}% to {val_accuracy:.2f}%. Saving model...')
            best_val_accuracy = val_accuracy
            save_model(model, config['model_save_path'])
    
    print('Training complete!')
    print(f'Best validation accuracy: {best_val_accuracy:.2f}%')
    
    # load the best model and evaluate
    model = load_model(model, config['model_save_path'])
    evaluate_model(model, val_loader, device)

def validate(model, val_loader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    
    return accuracy, avg_loss

def evaluate_model(model, val_loader, device):
    """Evaluate the model and print classification report."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

def save_model(model, path):
    """Save the model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load the model from disk."""
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

if __name__ == '__main__':
    train_model()