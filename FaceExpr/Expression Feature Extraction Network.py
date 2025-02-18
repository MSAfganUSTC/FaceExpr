import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Linear(1, dim)
    
    def forward(self, t):
        return self.embedding(t.view(-1, 1))

# Attention Mechanisms
class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
    
    def forward(self, x):
        return self.attention(x)

# Expression Feature Extraction Network
class ExpressionFeatureExtractor(nn.Module):
    def __init__(self, unet_encoder, unet_decoder, r_net, input_dim=512):
        super().__init__()
        self.temporal_emb = TemporalEmbedding(input_dim)
        self.unet_encoder = unet_encoder
        self.unet_decoder = unet_decoder
        self.r_net = r_net
        self.p_atten = AttentionModule(input_dim)
        self.c_atten = AttentionModule(input_dim)
    
    def forward(self, I, P_exp, T_steps):
        T_emb = self.temporal_emb(T_steps)
        extracted_features = []
        
        for t in range(T_steps, 0, -1):
            T_emb_t = self.temporal_emb(torch.tensor([t], dtype=torch.float32))
            
            for i in range(1, 5):
                if i in [2, 3] and (0.3 * T_steps <= t <= 0.7 * T_steps):
                    f_exp_n = self.unet_encoder(self.r_net(self.p_atten(I), P_exp), T_emb_t)
                
                X = self.unet_encoder(self.r_net(self.c_atten(I), P_exp), T_emb_t)
                extracted_features.append(X)  # Store extracted features from downsampling blocks
            
            # Feature fusion from downsampling blocks
            fused_features = torch.cat(extracted_features, dim=1)
            
            for i in range(1, 5):
                Y = self.unet_decoder(self.r_net(self.c_atten(fused_features), P_exp), T_emb_t)
        
        return I, f_exp_n, fused_features

# Example usage
if __name__ == "__main__":
    input_image = torch.randn(1, 3, 256, 256)
    P_exp = torch.randn(1, 512)
    T_steps = 10
    
    # Dummy U-Net encoder, decoder, and R-Net
    unet_encoder = nn.Identity()
    unet_decoder = nn.Identity()
    r_net = nn.Identity()
    
    model = ExpressionFeatureExtractor(unet_encoder, unet_decoder, r_net)
    I_ref, f_exp_n, fused_features = model(input_image, P_exp, T_steps)
    print(f"Extracted feature shape: {fused_features.shape}")
