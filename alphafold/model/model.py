import torch

from alphafold.model.evoformer import EvoformerStack
from alphafold.model.extra_msa_stack import ExtraMsaEmbedder, ExtraMsaStack
from alphafold.model.input_embedder import InputEmbedder
from alphafold.model.recycling_embedder import RecyclingEmbedder
from alphafold.model.structure_module import StructureModule


class Model(torch.nn.Module):
    """
    Implementation of the Algorithm 2.
    """

    def __init__(self,
                 c_m: int = 256,
                 c_z: int = 128,
                 c_e: int = 64,
                 f_e: int = 25,
                 tf_dim: int = 21,
                 c_s: int = 384,
                 num_blocks_extra_msa: int = 4,
                 num_blocks_evoformer: int = 48,
                 num_layers_structure: int = 8,
                 c_a: int = 128):
        """
        Initializes the Algorithm 2.

        Args:
            c_m:    Dimension for the MSA representation.
            c_z:    Dimension for the pairwise representation.
            c_e:    Dimension for the extra MSA representation.
            f_e:    Dimension for the extra MSA features.
            tf_dim: Dimension for the target features.
            c_s:    Dimension for the single representation.
            num_blocks_extra_msa:   Number of extra MSA blocks.
            num_blocks_evoformer:   Number of evoformer blocks.
            c_a:    Embedding dimension for the AngleResNet
            num_blocks_structure:   Number of layers for the structure module.
        """

        self.c_m = c_m
        self.c_z = c_z

        super().__init__()

        self.input_embedder = InputEmbedder(c_m, c_z, tf_dim)
        self.extra_msa_embedder = ExtraMsaEmbedder(f_e, c_e)
        self.recycling_embedder = RecyclingEmbedder(c_m, c_z)
        self.extra_msa_stack = ExtraMsaStack(c_e,
                                             c_z,
                                             num_blocks=num_blocks_extra_msa)
        self.evoformer = EvoformerStack(c_m,
                                        c_z,
                                        num_blocks=num_blocks_evoformer,
                                        c_s=c_s)
        self.structure_module = StructureModule(c_s,
                                                c_z,
                                                n_layer=num_layers_structure,
                                                c=c_a)
        self.input_embedder = InputEmbedder(c_m, c_z, tf_dim)
        self.extra_msa_embedder = ExtraMsaEmbedder(f_e, c_e)
        self.recycling_embedder = RecyclingEmbedder(c_m, c_z)
        self.extra_msa_stack = ExtraMsaStack(c_e,
                                             c_z,
                                             num_blocks=num_blocks_extra_msa)
        self.evoformer = EvoformerStack(c_m,
                                        c_z,
                                        num_blocks=num_blocks_evoformer)
        self.structure_module = StructureModule(c_s,
                                                c_z,
                                                n_layer=num_layers_structure,
                                                c=c_a)

    def forward(self, batch):
        """
        Forward pass for the Algorithm 2.

        Args:
            batch:  A dictionary that contains:
                * msa_feat: A PyTorch tensor with a shape of
                            (*, N_seq, N_res, msa_feat_dim, N_cycle).
                * extra_msa_feat:   A PyTorch tensor with a shape of
                                    (*, N_extra, N_res, f_e, N_cycle).
                * target_feat:  A PyTorch tensor with a shape of
                                (*, N_res, tf_dim, N_cycle) which contains the
                                one-hot encoding of the target sequence.
                * residue_index:    A PyTorch tensor with a shape of
                                    (*, N_res, N_cycle) which contains the
                                    aminoacid indices.

        Returns:
            A dictionary that contains:
                * final_positions:  A PyTorch tensor with a shape of
                                    (*, N_res, 37, 3, N_cycle) that contains
                                    the positions of each atom.
                * position_mask:    A PyTorch tensor with a shape of
                                    (*, N_res_ 37, N_cycle) that contains the
                                    non included aminoacid for each residue.
                * angles:   A PyTorch tensor with a shape of
                            (*, N_layers, N_res, n_torsion_angles, 2, N_cycle)
                            that contains the torsion angles. 
                * frames:   A PyTorch tensor with a shape of
                            (*, N_layers, N_res, 4, 4, N_cycle).
        """
        outputs = {}

        N_cycle = batch["msa_feat"].shape[-1]
        N_seq, N_res = batch["msa_feat"].shape[-4:-2]
        batch_shape = batch["msa_feat"].shape[:-4]

        device = batch["msa_feat"].device
        dtype = batch["msa_feat"].dtype

        m_prev = torch.zeros(batch_shape + (N_seq, N_res, self.c_m),
                             device=device,
                             dtype=dtype)
        z_prev = torch.zeros(batch_shape + (N_res, N_res, self.c_z),
                             device=device,
                             dtype=dtype)
        pseudo_beta_x_prev = torch.zeros(batch_shape + (N_res, 3),
                                         device=device,
                                         dtype=dtype)

        for i in range(N_cycle):
            print(f"Starting cycle {i}")
            current_batch = {k: v[..., i] for k, v in batch.items()}

            m, z = self.input_embedder(current_batch)

            m_rec, z_rec = self.recycling_embedder(m_prev, z_prev,
                                                   pseudo_beta_x_prev)

            m[..., 0, :, :] += m_rec
            z += z_rec

            e = self.extra_msa_embedder(current_batch)
            z = self.extra_msa_stack(e, z)
            del e

            m, z, s = self.evoformer(m, z)

            F = current_batch["target_feat"].argmax(dim=-1)
            # F = current_batch["residue_index"]

            structure_out = self.structure_module(s, z, F)
            m_prev = m
            z_prev = z
            pseudo_beta_x_prev = structure_out["pseudo_beta_positions"]

            for k, v in structure_out.items():
                if k in outputs:
                    outputs[k].append(v)
                else:
                    outputs[k] = [v]

        outputs = {k: torch.stack(v, dim=-1) for k, v in outputs.items()}
        return outputs
