import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    """Attention."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
    ) -> None:
        """Initialize the module."""

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads 

        self.linear_qkv = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_out = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        q, k, v = rearrange(self.linear_qkv(x), 'b s (k h e) -> k b h s e', k=3, h=self.number_of_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.linear_out(rearrange(x, 'b h s e -> b s (h e)'))
      
        return x


class ResidualBlock(nn.Module):
    """Residual block."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
    ) -> None:
        """Initialize the module."""

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )

        self.normalization_1 = nn.LayerNorm(normalized_shape=embedding_dimension)
        self.normalization_2 = nn.LayerNorm(normalized_shape=embedding_dimension)
        
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 3,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=embedding_dimension * 3,
                out_features=embedding_dimension,
            ),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        x = x + self.attention(self.normalization_1(x), mask)
        x = x + self.mlp(self.normalization_2(x))

        return x
    

class Transformer(nn.Module):
    """Transformer."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        vocabulary_size: int,
        maximum_sequence_length: int,
    ) -> None:
        """Initialize the module."""

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_layers = number_of_layers
        self.vocabulary_size = vocabulary_size
        self.maximum_sequence_length = maximum_sequence_length

        self.position_embedding = nn.Embedding(
            num_embeddings=maximum_sequence_length,
            embedding_dim=embedding_dimension,
        )

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
            ) for _ in range(number_of_layers)
        ])

        self.token_unembedding = nn.Linear(
            in_features=embedding_dimension,
            out_features=vocabulary_size,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Assuming x has shape (B, S, E).

        position = torch.arange(x.size(1))
        z = self.token_embedding(x) + self.position_embedding(position)

        for i, residual_block in enumerate(self.residual_blocks):
            z = residual_block(z, mask)
        
        log_probability = F.log_softmax(self.token_unembedding(z), dim=-1)

        return log_probability
