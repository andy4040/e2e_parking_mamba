import torch
from torch import nn
from model.mamba_block import MambaEncoder


class MambaFeatureFusion(nn.Module):
    """
    Mamba-based feature fusion module (V1)
    Takes the output from FeatureFusion (B, L, 258) and processes it with Mamba
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Mamba encoder parameters
        d_model = cfg.tf_en_dim  # 258
        n_layers = getattr(cfg, 'mamba_layers', 4)
        d_state = getattr(cfg, 'mamba_d_state', 16)
        d_conv = getattr(cfg, 'mamba_d_conv', 4)
        expand = getattr(cfg, 'mamba_expand', 2)
        
        # Mamba encoder
        self.mamba_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Residual connection
        self.use_residual = getattr(cfg, 'mamba_use_residual', True)
        
        if not self.use_residual:
            self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, fuse_feature):
        """
        Args:
            fuse_feature: (B, L, 258) from FeatureFusion (Transformer)
        Returns:
            output: (B, L, 258) processed through Mamba
            
        """
        
        mamba_out = self.mamba_encoder(fuse_feature)

        
        
        if self.use_residual:
            output = fuse_feature + mamba_out
   
        else:
            output = self.projection(mamba_out)
        return output
