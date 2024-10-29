import torch


class InvariantPointAttention(torch.nn.Module):
    """
    Implementation of the Algorithm 22.

    The embeddings of q, k, and v are computed with a linear layer that uses
    bias contrary to the supplementary material as the official implementation
    does use bias.
    """

    def __init__(self,
                 c_s: int,
                 c_z: int,
                 n_query_points: int = 4,
                 n_point_values: int = 8,
                 N_head: int = 12,
                 c: int = 16):
        """
        Initializes the InvariantPointAttention.

        Args:
            c_s:    Dimension of the single representation.
            c_z:    Dimension of the pairwise representation.
            N_head: Number of heads that will be used in multi head attention.
            c:  Embedding dimension for each head.
            n_query_points: Number of query points for the invariant point
                            attention.
            n_point_values: Number of value points for the invariant point
                            attention.

        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.N_head = N_head
        self.c = c
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values

        self.linear_q = torch.nn.Linear(c_s, c * N_head, bias=True)
        self.linear_k = torch.nn.Linear(c_s, c * N_head, bias=True)
        self.linear_v = torch.nn.Linear(c_s, c * N_head, bias=True)

        self.linear_q_points = torch.nn.Linear(c_s,
                                               n_query_points * 3 * N_head,
                                               bias=True)
        self.linear_k_points = torch.nn.Linear(c_s,
                                               n_query_points * 3 * N_head,
                                               bias=True)
        self.linear_v_points = torch.nn.Linear(c_s,
                                               n_point_values * 3 * N_head,
                                               bias=True)

        self.linear_b = torch.nn.Linear(c_z, N_head)
        self.linear_out = torch.nn.Linear(
            c_z * N_head + c * N_head + n_point_values * 3 * N_head +
            n_point_values * N_head, c_s)

        # gamma
        self.head_weights = torch.nn.Parameter(torch.zeros((N_head, )))
        self.softplus = torch.nn.Softplus()

    def prepare_qkv(self, s: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Creates embeddings q, k, v and point embeddings qp, kp, vp.

        Args:
            s:  A PyTorch tensor with a shape of (*, N_res, c_s) thar includes
                the single representation.

        Returns:
            A tuple of PyTorch tensors q, k, v with a shape of
            (*, N_head, N_res, c) and qp, kp, vp with a shape of
            (*, N_head, n_query_points, N_res, 3).
        """

        # (*, N_res, c_s) -> (*, N_res, c * N_head)
        q = self.linear_q(s)
        k = self.linear_k(s)
        v = self.linear_v(s)

        # (*, N_res, c_s) -> (*, N_res, n_query_points * 3 * N_head)
        qp = self.linear_q_points(s)
        kp = self.linear_k_points(s)
        vp = self.linear_v_points(s)

        # (*, N_res, c * N_head) -> (*, N_res, N_head, c)
        q = q.view(q.shape[:-1] + (self.N_head, self.c))
        k = k.view(k.shape[:-1] + (self.N_head, self.c))
        v = v.view(v.shape[:-1] + (self.N_head, self.c))

        # (*, N_res, n_query_points * 3 * N_head) -> (*, N_res,3, N_head, n_query_points)
        qp = qp.view(qp.shape[:-1] + (3, self.N_head, self.n_query_points))
        kp = kp.view(kp.shape[:-1] + (3, self.N_head, self.n_query_points))
        vp = vp.view(vp.shape[:-1] + (3, self.N_head, self.n_point_values))

        # (*, N_res, N_head, c) -> (*, N_head, N_res, c)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        # (*, N_res, 3, N_head, n_query_points) -> (*, N_head, n_query_points, N_res, 3)
        print("a", qp.shape)
        qp = qp.movedim(-3, -1).movedim(-4, -2)
        kp = kp.movedim(-3, -1).movedim(-4, -2)
        vp = vp.movedim(-3, -1).movedim(-4, -2)
        print("b", qp.shape)

        return q, k, v, qp, kp, vp
