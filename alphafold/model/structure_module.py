import torch

from alphafold.utils.geometry import quat_to_3x3_rotation, assemble_4x4_transform


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
