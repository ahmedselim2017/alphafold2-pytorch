import torch


class InputEmbedder(torch.nn.Module):
    """
    Implementation of the Algorithm 3 and Algorithm 4.
    """

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 tf_dim: int,
                 msa_feat_dim: int = 49,
                 vbins: int = 32):
        """
        Initializes the InputEmbedder.

        Args:
            c_m:    Embedding dimension of the MSA representation.
            c_z:    Embedding dimension of the pairwise representation.
            tf_dim: Embedding dimension of target_feat.
            msa_feat_dim:   Embedding dimension of the MSA feature.
            vbins:  Determines the bins for relpos as
                    (-vbins, -vbins+1,...,vbins).
        """

        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins

        self.linear_tf_z_i = torch.nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = torch.nn.Linear(tf_dim, c_z)
        self.linear_tf_m = torch.nn.Linear(tf_dim, c_m)
        self.linear_msa_m = torch.nn.Linear(msa_feat_dim, c_m)
        self.linear_relpos = torch.nn.Linear(2 * vbins + 1, c_z)

    def relpos(self, residue_index: torch.Tensor):
        """
        Implementation of the Algorithm 4.

        Args:
            residue_index:  A PyTorch tensor that includes indices for the
                            aminoacids in the sequence, which can be written as
                            [0,...N_res-1].

        Returns:
            A PyTorch tensor of shape (N_res, N_res, 2 * vbins + 1) that
            includes the relpo encoded residue_index.
        """

        dtype = self.linear_relpos.weight.dtype

        residue_index = residue_index.long()

        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        d = torch.clamp(d, -self.vbins, self.vbins) + self.vbins

        d_onehot = torch.nn.functional.one_hot(d,
                                               num_classes=2 * self.vbins +
                                               1).to(dtype=dtype)

        return self.linear_relpos(d_onehot)
