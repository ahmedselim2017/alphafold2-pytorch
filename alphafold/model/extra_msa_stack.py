from alphafold.model.multi_head_attention import MultiHeadAttention
from alphafold.model.dropout import DropoutRowwise, DropoutColumnwise
from alphafold.model.msa_stack import MSARowAttentionWithPairBias, MSATransition, OuterProductMean
import torch

from alphafold.model.pair_stack import PairStack


class ExtraMSAEmbedder(torch.nn.Module):
    """
    Implementation of the extra MSA embedder.
    """

    def __init__(self, c_f: int, c_e: int):
        """
        Initializes the ExtraMSAEmbedder.

        Args:
            c_f:    Dimension of the extra_msa_feat.
            c_e:    Embedding dimension of the extra_msa_feat.
        """

        super().__init__()

        self.linear = torch.nn.Linear(c_f, c_e)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass for the ExtraMSAEmbedder.

        Args:
            batch:  Feature dictionary with the following entries:
                        * extra_msa_feat: Extra MSA feature of shape
                          (*, N_extra, N_res, c_f).

        Returns:
            Output PyTorch tensor of shape (*, N_extra, N_res, c_e)
        """

        return self.linear(batch["extra_msa_feat"])


class MSAColumnGlobalAttention(torch.nn.Module):
    """
    Implementation of the Algorithm 19.
    """

    def __init__(self, c_m: int, c: int = 8, N_head: int = 8):
        """
        Initializes the MsaColumnGlobalAttention.

        Args:
            c_m:    Embedding dimension of the MSA representation.
            c:  Embedding dimension for MultiHeadAttention.
            N_head: Number of heads for MultiHeadAttention.
        """

        super().__init__()

        self.layer_norm_m = torch.nn.LayerNorm(c_m)
        self.global_attention = MultiHeadAttention(c_m,
                                                   c,
                                                   N_head=N_head,
                                                   attn_dim=-3,
                                                   gated=True,
                                                   is_global=True)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the MsaColumnGlobalAttention.

            Args:
                m:  MSA representation of shape (*, N_seq, N_res, c_m).

            Returns:
                A PyTorch tensor with the shape of m.
        """

        m = self.layer_norm_m(m)
        m = self.global_attention(m)

        return m


class ExtraMSABlock(torch.nn.Module):
    """
    Implementation of one block from the Algorithm 18.
    """

    def __init__(self, c_e: int, c_z: int):
        """
        Initializes the extra MSA block.

        Args:
            c_e:    Embedding dimension of the extra MSA representation.
            c_z:    Embedding dimension of the pairwise representation.
        """

        super().__init__()

        self.dropout_rowwise = DropoutRowwise(p=0.15)
        self.msa_att_row = MSARowAttentionWithPairBias(c_e, c_z, c=8)
        self.msa_att_col = MSAColumnGlobalAttention(c_e)
        self.msa_transition = MSATransition(c_e)

        self.outer_product_mean = OuterProductMean(c_e, c_z)

        self.core = PairStack(c_z)

    def forward(self, e: torch.Tensor,
                z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the extra MSA block.

        Args:
            e:  A PyTorch tensor with a shape of (*, N_extra, N_res, c_e) that
                contains the extra MSA representation.
            z:  A PyTorch tensor with a shape of (*, N_res, N_res, c_e) that
                contains the pairwise representation.

        Returns:
            Two PyTorch tensors with the shapes of e and z.
        """

        e = e + self.dropout_rowwise(self.msa_att_row(e, z))
        e = e + self.msa_att_col(e)
        e = e + self.msa_transition(e)

        z = z + self.outer_product_mean(e)

        z = self.core(z)

        return e, z


class ExtraMSAStack(torch.nn.Module):
    """
    Implementation of the Algorithm 18.
    """

    def __init__(self, c_e: int, c_z: int, N_block: int = 4):
        """
        Initializes the extra MSA stack.

        Args:
            c_e:    Embedding dimension of the extra MSA representation.
            c_z:    Embedding dimension of the pairwise representation.
            N_block:    Number of ExtraMSABlocks that should be used
        """

        super().__init__()

        self.blocks = torch.nn.ModuleList(
            [ExtraMSABlock(c_e, c_z) for _ in range(N_block)])

    def forward(self, e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the extra MSA stack.

        Args:
            e:  A PyTorch tensor with a shape of (*, N_extra, N_res, c_e) that
                contains the extra MSA representation.
            z:  A PyTorch tensor with a shape of (*, N_res, N_res, c_e) that
                contains the pairwise representation.
        Returns:
            A PyTorch tensor with the shape of z.
        """

        for block in self.blocks:
            e, z = block(e, z)

        return z
