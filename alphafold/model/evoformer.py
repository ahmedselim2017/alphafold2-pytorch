import torch
from alphafold.model.dropout import DropoutRowwise
from alphafold.model.msa_stack import (MSARowAttentionWithPairBias,
                                       MSAColumnAttention, MSATransition,
                                       OuterProductMean)

from alphafold.model.pair_stack import PairStack


class EvoformerBlock(torch.nn.Module):
    """
    Implementation of one Evoformer block from the Algorithm 6.
    """

    def __init__(self, c_m: int, c_z: int):
        """
        Initializes the EvoformerBlock.

        Args:
            c_m:    Embedding dimension for the MSA representation.
            c_z:    Embedding dimension for the pairwise representation.
        """

        super().__init__()

        self.dropout_rowwise_m = DropoutRowwise(p=0.15)
        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z)
        self.msa_att_col = MSAColumnAttention(c_m)
        self.msa_transition = MSATransition(c_m)
        self.outer_product_mean = OuterProductMean(c_m, c_z)
        self.core = PairStack(c_z)

    def forward(self, m: torch.Tensor,
                z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Evoformer block.

        Args:
            m:  A PyTorch tensor of shape (*, N_seq, N_res, c_m) that contains
                the MSA representation.
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_m) that contains
                the pairwise representation.

        Returns:
            PyTorch tensors with the shapes of m and z.
        """

        m = m + self.dropout_rowwise_m(self.msa_att_row(m, z))
        m = m + self.msa_att_col(m)
        m = m + self.msa_transition(m)

        z = z + self.outer_product_mean(m)

        z = self.core(z)

        return m, z


class EvoformerStack(torch.nn.Module):
    """
    Implementation of the Evoformer stack from the Algorithm 6.
    """

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 num_blocks: int = 48,
                 c_s: int = 384):
        """
        Initializes the Evoformer stack.

        Args:
            c_m:    Embedding dimension for the MSA representation.
            c_z:    Embedding dimension for the pairwise representation.
                    num_blocks: Number of Evoformer blocks that should be used.
            c_s:    Number of channels that should be used for the single
                    representation.
        """

        super().__init__()

        self.blocks = torch.nn.ModuleList(
            [EvoformerBlock(c_m, c_z) for _ in range(num_blocks)])
        self.linear = torch.nn.Linear(c_m, c_s)

    def forward(
            self, m: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Evoformer stack.

        Args:
            m:  A PyTorch tensor of shape (*, N_seq, N_res, c_m) that contains
                the MSA representation.
            z:  A PyTorch tensor of shape (*, N_res, N_res, c_m) that contains
                the pairwise representation.

        Returns:
            Output tensors with the same shape as m, z and the single
            representation tensor with the shape of (*, N_res, c_s).

        """

        for evoformer_block in self.blocks:
            m, z = evoformer_block(m, z)
        s = self.linear(m[..., 0, :, :])

        return m, z, s
