import torch
import torch.nn as nn
import torch.nn.functional as F


class S6(nn.Module):
    """Simplified State Space Model (S6) - Core of Mamba with Parallel Scan"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolutional layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # State space matrices
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) where B=batch, L=sequence length, D=d_model
        Returns:
            output: (B, L, D)
        """
        batch, seqlen, dim = x.shape
       
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Convolutional processing
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seqlen]  # Trim padding
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # SSM processing
        x = F.silu(x)
        y = self.ssm(x)
        
        # Gating mechanism
        y = y * F.silu(z)
        
        # Output projection
    
        
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        """Simplified State Space Model computation"""
        batch, seqlen, d_inner = x.shape
        
        # Get SSM parameters
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()
        
        # Compute delta (time step) and B, C matrices
        deltaBC = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B = deltaBC.split([self.d_state, self.d_state], dim=-1)
        C = B  # Simplified: use same for B and C
        
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Discretize continuous parameters
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # SSM computation with PARALLEL SCAN
        y = self.parallel_scan(x, deltaA, deltaB, C, D)
        
        return y
    
    def parallel_scan(self, x, deltaA, deltaB, C, D):
        """
        Parallel scan implementation using associative scan
        Much faster than sequential scan!
        
        Args:
            x: (B, L, d_inner)
            deltaA: (B, L, d_inner, d_state)
            deltaB: (B, L, d_inner, d_state)
            C: (B, L, d_state)
            D: (d_inner,)
        Returns:
            y: (B, L, d_inner)
        """
        batch, seqlen, d_inner = x.shape
        d_state = deltaA.shape[-1]
        
        # Prepare for parallel scan
        # We'll use the fact that SSM can be written as:
        # h[t] = A[t] * h[t-1] + B[t] * x[t]
        # This is an associative operation!
        
        # Reshape for easier manipulation
        # (B, L, d_inner, d_state)
        As = deltaA
        Bs = deltaB * x.unsqueeze(-1)  # (B, L, d_inner, d_state)
        
        # Binary associative scan
        # This reduces O(L) sequential to O(log L) depth
        h = self._associative_scan(As, Bs)
        
        # Compute output: y = C * h + D * x
        # h: (B, L, d_inner, d_state)
        # C: (B, L, d_state)
        y = torch.einsum('blnd,bld->bln', h, C) + D * x
        
        return y
    
    def _associative_scan(self, As, Bs):
        """
        Associative scan using parallel prefix sum
        
        For SSM: (A, B) ⊕ (A', B') = (A' * A, A' * B + B')
        
        Args:
            As: (B, L, d_inner, d_state) - transition matrices
            Bs: (B, L, d_inner, d_state) - input contributions
        Returns:
            h: (B, L, d_inner, d_state) - hidden states
        """
        B, L, d_inner, d_state = As.shape
        
        # We'll do a tree-based parallel scan
        # Complexity: O(log L) depth instead of O(L)
        
        # Check if length is power of 2, if not, pad
        log_L = (L - 1).bit_length()
        padded_L = 2 ** log_L
        
        if padded_L > L:
            # Pad with identity elements
            pad_size = padded_L - L
            # Identity: A=1, B=0
            As = torch.cat([
                As,
                torch.ones(B, pad_size, d_inner, d_state, device=As.device, dtype=As.dtype)
            ], dim=1)
            Bs = torch.cat([
                Bs,
                torch.zeros(B, pad_size, d_inner, d_state, device=Bs.device, dtype=Bs.dtype)
            ], dim=1)
        
        # Up-sweep (reduce) phase
        current_As = As.clone()
        current_Bs = Bs.clone()
        
        for d in range(log_L):
            stride = 2 ** (d + 1)
            half_stride = 2 ** d
            
            # Indices to update
            indices = torch.arange(stride - 1, padded_L, stride, device=As.device)
            
            if len(indices) == 0:
                continue
            
            # Get pairs: (left, right)
            left_idx = indices - half_stride
            right_idx = indices
            
            # Combine: (A[right] * A[left], A[right] * B[left] + B[right])
            new_A = current_As[:, right_idx] * current_As[:, left_idx]
            new_B = current_As[:, right_idx] * current_Bs[:, left_idx] + current_Bs[:, right_idx]
            
            current_As[:, right_idx] = new_A
            current_Bs[:, right_idx] = new_B
        
        # Down-sweep phase
        # Initialize result with final values
        h = torch.zeros_like(Bs)
        h[:, padded_L - 1] = current_Bs[:, padded_L - 1]
        
        for d in range(log_L - 1, -1, -1):
            stride = 2 ** (d + 1)
            half_stride = 2 ** d
            
            # Indices to update
            left_indices = torch.arange(half_stride - 1, padded_L, stride, device=As.device)
            
            if len(left_indices) == 0:
                continue
            
            right_indices = left_indices + half_stride
            
            # Filter valid indices
            valid_mask = right_indices < padded_L
            left_indices = left_indices[valid_mask]
            right_indices = right_indices[valid_mask]
            
            if len(left_indices) == 0:
                continue
            
            # Propagate from right to left
            # h[left] = A[left] * h[parent] + B[left]
            # But we need to be careful about the parent index
            
            # Actually, in down-sweep, we propagate accumulated results
            parent_indices = ((left_indices + 1) // stride) * stride - 1
            
            h[:, left_indices] = (
                current_As[:, left_indices] * h[:, parent_indices] + 
                current_Bs[:, left_indices]
            )
            
            # Update right if needed
            if torch.any(right_indices < padded_L):
                valid_right = right_indices < padded_L
                valid_right_idx = right_indices[valid_right]
                valid_left_idx = left_indices[valid_right]
                
                h[:, valid_right_idx] = (
                    current_As[:, valid_right_idx] * h[:, valid_left_idx] + 
                    current_Bs[:, valid_right_idx]
                )
        
        # Remove padding if any
        if padded_L > L:
            h = h[:, :L]
        
        return h


class MambaBlock(nn.Module):
    """Mamba block with residual connection and normalization"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = S6(d_model, d_state, d_conv, expand)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        return x + self.mamba(self.norm(x))


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks for feature processing"""
    
    def __init__(self, d_model, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) - fused features from FeatureFusion
        Returns:
            output: (B, L, D) - processed features
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


if __name__ == "__main__":
    # Test the implementation

    
    batch_size = 4
    seq_len = 256
    d_model = 258
    
    # Create model
    model = MambaEncoder(d_model=d_model, n_layers=4)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
   
    
    # Forward pass
    import time
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Benchmark
    n_runs = 10
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            output = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / n_runs * 1000  # ms
    
 