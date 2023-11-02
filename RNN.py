import torch
import torch.nn as nn

# Define the RNN model
class Classifier(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, output_size: int, num_layers: int, vocab_size: int):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = torch.sigmoid(self.fc(output[:, -1, :]))  
        return output

def train(model: Classifier, optimizer, loss_fn, train_iter, val_iter, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_iter:
            text, labels = batch.text, batch.label
            optimizer.zero_grad()

            outputs = model(text)
            loss = loss_fn(outputs, labels.view(-1, 1)) 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        
        average_loss = total_loss / total_samples
        
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_iter:
                text, labels = batch.text, batch.label
                outputs = model(text)
                predictions = (outputs > 0.5).float() 
                total_correct += (predictions == labels.view_as(predictions)).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.2%}, Validation Accuracy: {accuracy:.2%}')
