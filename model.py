import torch
import torch.nn as nn
from decoder import TemporalCNNDecoder


class ImageCaptioningModel(nn.Module):
    """CNN encoder + decoder."""
    def __init__(self, encoder, decoder, encoder_feature_dim, decoder_hidden_dim):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.feature_proj = nn.Linear(encoder_feature_dim, decoder_hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, images, captions):
        """Forward pass."""
        # encode image
        encoder_features = self.encoder(images)
        
        # pool spatial features
        if encoder_features.dim() == 3:
            encoder_features = encoder_features.mean(dim=1)
        
        # project features
        encoder_features = self.feature_proj(encoder_features)
        encoder_features = self.relu(encoder_features)
        
        # decode captions
        logits = self.decoder(captions, encoder_features)
        
        return logits

    def generate_caption(self, image, start_token_id, end_token_id, 
                         max_length=30, temperature=1.0):
        """Generate single caption."""
        self.eval()
        
        # add batch dim
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            # encode image
            encoder_features = self.encoder(image)
            
            # pool features
            if encoder_features.dim() == 3:
                encoder_features = encoder_features.mean(dim=1)
            
            encoder_features = self.feature_proj(encoder_features)
            encoder_features = self.relu(encoder_features)
            
            # generate tokens
            generated_ids = self.decoder.generate(
                start_token=start_token_id,
                max_length=max_length,
                encoder_features=encoder_features,
                temperature=temperature
            )
        
        return generated_ids


def build_model(vocab_size, embed_dim=512, decoder_hidden_dim=512, 
                num_decoder_layers=6, encoder_feature_dim=2048, 
                kernel_size=3, dropout=0.3):
    """Build full model."""
    # import encoder
    try:
        from encoder import EncoderCNN
        encoder = EncoderCNN(embed_size=embed_dim)
    except ImportError:
        # fallback placeholder
        print("Warning: Using placeholder encoder. Person 1 needs to implement encoder.py")
        encoder = PlaceholderEncoder(encoder_feature_dim)
    
    # init decoder
    decoder = TemporalCNNDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=decoder_hidden_dim,
        num_layers=num_decoder_layers,
        kernel_size=kernel_size,
        dropout=dropout
    )
    
    # wrap model
    model = ImageCaptioningModel(
        encoder=encoder,
        decoder=decoder,
        encoder_feature_dim=embed_dim,
        decoder_hidden_dim=decoder_hidden_dim
    )
    
    return model


class PlaceholderEncoder(nn.Module):
    """Placeholder encoder."""
    def __init__(self, feature_dim=2048):
        super().__init__()
        # simple CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Test the model
    vocab_size = 5000
    model = build_model(vocab_size)
    
    # Dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 20))
    
    # Forward pass
    logits = model(images, captions)
    print(f"Input images: {images.shape}")
    print(f"Input captions: {captions.shape}")
    print(f"Output logits: {logits.shape}")
    print("Model test passed!")

