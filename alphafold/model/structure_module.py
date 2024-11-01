import torch

from alphafold.model.invariant_point_attention import InvariantPointAttention
from alphafold.utils import residue_constants
from alphafold.utils.geometry import compute_all_atom_coordinates, quat_to_3x3_rotation, assemble_4x4_transform


class StructureModuleTransition(torch.nn.Module):
    """
    Implementation of the lines 8 and 9 in the Algorithm 20.
    """

    def __init__(self, c_s: int):
        """
        Initializes the StructureModuleTransition.

        Args:
            c_s:    Dimension of the single representation.
        """

        super().__init__()

        self.linear_1 = torch.nn.Linear(c_s, c_s)
        self.linear_2 = torch.nn.Linear(c_s, c_s)
        self.linear_3 = torch.nn.Linear(c_s, c_s)
        self.layer_norm = torch.nn.LayerNorm(c_s)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the StructureModuleTransition.

        Args:
            s:  A PyTorch tensor with a shape of (*, N_res, c_s) that includes
                the single representation.

        Returns:
            A PyTorch tensor with the same shape as s.
        """

        s = s + self.linear_3(
            self.relu(self.linear_2(self.relu(self.linear_1(s)))))
        s = self.layer_norm(self.dropout(s))
        return s


class BackboneUpdate(torch.nn.Module):
    """
    Implementation of the Algorithm 23.
    """

    def __init__(self, c_s: int):
        """
        Initializes the BackboneUpdate.

        Args:
            c_s:    Dimension of the single representation.
        """

        super().__init__()

        self.linear = torch.nn.Linear(c_s, 6)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BackboneUpdate.

        Args:
            s:  A PyTorch tensor with a shape of (*, N_res, c_s) that includes
                the single representation.
        """

        outs = self.linear(s)

        quat_ones = torch.ones(s.shape[:-1] + (1, ),
                               device=s.device,
                               dtype=s.dtype)

        quat = torch.cat((quat_ones, outs[..., :3]), dim=-1)
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)

        R = quat_to_3x3_rotation(quat)
        t = outs[..., 3:]

        return assemble_4x4_transform(R, t)


class AngleResNetLayer(torch.nn.Module):
    """
    Implementation of the angle ResNet layer which is used in the lines 12 and
    13 in the Algorithm 20.
    """

    def __init__(self, c: int):
        """
        Initializes the AngleResNetLayer.

        Args:
            c:  Embedding dimension for the Angle ResNet.
        """

        super().__init__()

        self.linear_1 = torch.nn.Linear(c, c)
        self.linear_2 = torch.nn.Linear(c, c)
        self.relu = torch.nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AngleResNetLayer.

        Args:
            a: Embedding with a shape of (*, N_res, c).

        Returns:
            A PyTorch tensor with the same shape as a.
        """

        a = a + self.linear_2(self.relu(self.linear_1(self.relu(a))))

        return a


class AngleResNet(torch.nn.Module):
    """
    Implementation of the angle ResNet from lines 11 to 14 in the Algorithm 20.
    """

    def __init__(self,
                 c_s: int,
                 c: int = 128,
                 n_torsion_angles: int = 7,
                 n_layer: int = 2):
        """
        Initializes the anlge ResNet.

        Args:
            c_s:    Dimension of the single representation.
            c:  Embedding dimension for the AngleResNet.
            N_layer:    Number of AngleResNetLayer that should be used.
        """

        super().__init__()

        self.n_torsion_angles = n_torsion_angles

        self.linear_in = torch.nn.Linear(c_s, c)
        self.linear_initial = torch.nn.Linear(c_s, c)
        self.layers = torch.nn.ModuleList(
            [AngleResNetLayer(c) for _ in range(n_layer)])
        # Twice the number of torsion angles as cosine and sine of the angles
        # are being predicted
        self.linear_out = torch.nn.Linear(c, 2 * n_torsion_angles)
        self.relu = torch.nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor):
        """
        Forward pass for the angle ResNet.

        Uses ReLU to the single and initial single representations while
        generating the embedding contrary to the supplementary information
        as the official implementation also uses ReLU.

        Args:
            s:  A PyTorch tensor with a shape of (*, N_res, c) that contains
                the single representation
            s_initial:  A PyTorch tensor with a shape of (*, N_res, c) that
                        contains the initial single representation
        
        Returns:
            A PyTorch tensor with a shape of (*, N_res, n_torsion_angles, 2)
        """

        a = self.linear_in(self.relu(s)) + self.linear_initial(
            self.relu(s_initial))
        for layer in self.layers:
            a = layer(a)

        alpha = self.linear_out(self.relu(a))
        alpha_shape = alpha.shape[:-1] + (self.n_torsion_angles, 2)
        alpha = alpha.view(alpha_shape)

        return alpha


class StructureModule(torch.nn.Module):
    """
    Implementation of the Algorithm 20.
    """

    def __init__(self, c_s: int, c_z: int, n_layer: int = 8, c: int = 128):
        """
        Initializes the StructureModule.

        Args:
            c_s:    Dimension of the single representation.
            c_s:    Dimension of the pairwise representation.
            n_layer:    Number of layers for the StructureModule.
            c:  Embedding dimension for the AngleResNet.
        """

        super().__init__()

        self.n_layer = n_layer

        self.layer_norm_s = torch.nn.LayerNorm(c_s)
        self.layer_norm_z = torch.nn.LayerNorm(c_z)
        self.linear_in = torch.nn.Linear(c_s, c_s)
        self.layer_norm_ipa = torch.nn.LayerNorm(c_s)
        self.dropout_s = torch.nn.Dropout(p=0.1)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.transition = StructureModuleTransition(c_s)
        self.bb_update = BackboneUpdate(c_s)
        self.angle_resnet = AngleResNet(c_s, c=c)

    def process_outputs(self, T: torch.Tensor, alpha: torch.Tensor,
                        F: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Computes the final atom coordiantes, atom mask, and pseudo beta
        atom positions.

        Args:
            T:  A PyTorch tensor with a shape of (*, N_res, 4, 4) which includes
                the backbone transforms.
            alpha:  A PyTorch tensor with a shape of
                    (*, N_res, n_torsion_angles, 2) which includes the
                    torsion angles.
            F:  A PyTorch tensor with a shape of (*, N_res) which includes the
                aminoacid indices.

        Returns:
            A tuple that includes:
                * final_atom_positions: A PyTorch tensor with a shape of
                                        (*, N_res, 37, 3) which includes the
                                        final atom positions in Angstrom.
                * positions_mask:   A PyTorch tensor with a shape of
                                    (*, N_res, 37) which includes the non
                                    included aminoacid for each residue.
                * pseudo_beta_positions:    Positions of the pseudo beta
                                            atoms.
        """

        T_angstrom = T.clone()

        T_angstrom[..., :3, 3] *= 10
        final_atom_positions, position_mask = compute_all_atom_coordinates(
            T_angstrom, alpha, F)

        ca_ind = residue_constants.atom_types.index("CA")
        cb_ind = residue_constants.atom_types.index("CB")
        glycine_ind = residue_constants.restypes.index("G")

        pseudo_beta_positions = final_atom_positions[..., cb_ind, :]
        a_positions = final_atom_positions[..., ca_ind, :]
        pseudo_beta_positions[F == glycine_ind] = a_positions[F == glycine_ind]

        return final_atom_positions, position_mask, pseudo_beta_positions

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                F: torch.Tensor) -> dict:
        """
        Forward pass for the StructureModule.

        Args:
            s:  A PyTorch tensor with a shape of (*, N_res, c) that contains
                the single representation
            z:  A PyTorch tensor with a shape of (*, N_res, N_res) that
                contains the pairwise representation
            F:  A PyTorch tensor with a shape of (*, N_res) which includes the
                aminoacid indices.
        Returns:
            A dictionary containing:
                * final_atom_positions: A PyTorch tensor with a shape of
                                        (*, N_res, 37, 3) which includes the
                                        final atom positions in Angstrom.
                * position_mask:    A PyTorch tensor with a shape of
                                    (*, N_res, 37) which includes the non
                                    included aminoacid for each residue.
                * pseudo_beta_positions:    Positions of the pseudo beta
                                            atoms.
        """

        outputs: dict[str, list | torch.Tensor] = {"angles": [], "frames": []}

        N_res = s.shape[-2]
        batch_dim = s.shape[:-2]

        s_initial = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s = self.linear_in(s_initial)

        T = torch.eye(4, device=s.device, dtype=s.dtype)
        # (4, 4) -> (*, N_res, 4, 4)
        T = T.broadcast_to(batch_dim + (N_res, 4, 4))

        alpha = None
        for _ in range(self.n_layer):
            s = s + self.ipa(s, z, T)
            s = self.layer_norm_ipa(self.dropout_s(s))

            s = self.transition(s)

            T = T @ self.bb_update(s)
            outputs["frames"].append(T)  # type: ignore

            alpha = self.angle_resnet(s, s_initial)
            outputs["angles"].append(alpha)  # type: ignore

        outputs["frames"] = torch.stack(outputs["frames"], dim=-4)
        outputs["angles"] = torch.stack(outputs["angles"], dim=-4)

        assert alpha is not None
        final_atom_positions, position_mask, pseudo_beta_positions = self.process_outputs(
            T, alpha, F)

        outputs["final_positions"] = final_atom_positions
        outputs["position_mask"] = position_mask
        outputs["pseudo_beta_positions"] = pseudo_beta_positions

        return outputs
