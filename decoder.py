import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """Causal conv layer."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding = self.padding, dilation = dilation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)

        x = x[:, :, :-self.padding] if self.padding > 0 else x
        return self.dropout(x)

class TCNBlock(nn.Module):
    """TCN block residual."""
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation, dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)  # Residual connection
    
class TemporalCNNDecoder(nn.Module):
    """Temporal CNN decoder."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, 
                 kernel_size=3, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # embed to hidden
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        
        # encoder feature proj
        self.encoder_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # stacked TCN blocks
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # vocab output proj
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, captions, encoder_features=None):
        """Returns vocab logits."""
        batch_size, seq_len = captions.shape
        
        # embed tokens
        x = self.embedding(captions)
        x = self.embed_proj(x)
        
        # add image features
        if encoder_features is not None:
            img_features = self.encoder_proj(encoder_features)
            x = x + img_features.unsqueeze(1)
        
        # conv1d transpose
        x = x.transpose(1, 2)
        
        # run TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # transpose back
        x = x.transpose(1, 2)
        
        # project to vocab
        logits = self.output_proj(x)
        
        return logits

    def generate(self, start_token, max_length, encoder_features=None, temperature=1.0):
        """Greedy caption generation."""
        self.eval()
        generated = [start_token]
        
        with torch.no_grad():
            for _ in range(max_length):
                # build current seq
                current_seq = torch.LongTensor([generated]).to(next(self.parameters()).device)
                
                # get predictions
                logits = self.forward(current_seq, encoder_features)
                
                # last token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # pick most probable
                next_token = torch.argmax(next_token_logits).item()
                
                generated.append(next_token)
                
                # stop at END
                if next_token == 2:  
                    break
        
        return generated