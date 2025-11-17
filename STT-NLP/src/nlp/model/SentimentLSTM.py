import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    The LSTM model for sentiment analysis.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sig(out)
        
        out = out.view(batch_size, -1, self.output_dim)
        out = out[:, -1]
        
        return out