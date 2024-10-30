import torch
import math

from alphafold.utils.geometry import invert_4x4_transform, warp_3d_point


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
        qp = qp.movedim(-3, -1).movedim(-4, -2)
        kp = kp.movedim(-3, -1).movedim(-4, -2)
        vp = vp.movedim(-3, -1).movedim(-4, -2)

        return q, k, v, qp, kp, vp

    def compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor,
                                 qp: torch.Tensor, kp: torch.Tensor,
                                 z: torch.Tensor,
                                 T: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention scores for the invariant point attention.

        Args:
            q:  A PyTorch tensor with a shape of (*, N_head, N_res, c) for the
                query embeddings.
            k:  A PyTorch tensor with a shape of (*, N_head, N_res, c) for the
                key embeddings.
            qp: A PyTorch tensor with a shape of
                (*, N_head, n_query_points, N_res, 3) for the query point
                embeddings.
            kp: A PyTorch tensor with a shape of
                (*, N_head, n_query_points, N_res, 3) for the key point
                embeddings.
            z:  A PyTorch tensor with a shape of (*, N_res, N_res, c_z) that
                includes the pairwise representation.
            T:  A PyTorch tensor with a shape of (*, N_res, 4, 4) that includes
                the backbone transform.

        Returns:
            A PyTorch tensor with a shape of (*, N_head, N_res, N_res)
        """

        wc = math.sqrt(2 / (9 * self.n_query_points))
        wl = math.sqrt(1 / 3)

        gamma = self.softplus(self.head_weights).view((-1, 1, 1))

        # q = q / math.sqrt(self.c)

        # (*, N_res, N_res, c_z) -> (*, N_res, N_res, N_head)
        bias = self.linear_b(z)
        # (*, N_res, N_res, N_head) -> (*, N_head, N_res, N_res)
        bias = bias.movedim(-1, -3)

        qk = torch.einsum("...ic,...jc->...ij", q, k)

        # (*, N_res, 4, 4) -> (*, 1, 1, N_res, 4, 4)
        T_batch = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))

        # Unsqueeze to find pairwise point distances later
        T_qp = warp_3d_point(T_batch, qp).unsqueeze(-2)
        T_kp = warp_3d_point(T_batch, kp).unsqueeze(-3)

        # Find pairwise point distances
        sq_dist = torch.sum((T_qp - T_kp)**2, dim=-1)

        # Sum along points
        sq_dist_sum = torch.sum(sq_dist, dim=-3)

        extra = ((gamma * wc) / 2) * sq_dist_sum

        a = torch.softmax(wl * (qk / math.sqrt(self.c) + bias - extra), dim=-1)

        return a

    def compute_outputs(self, att_scores: torch.Tensor, z: torch.Tensor,
                        v: torch.Tensor, vp: torch.Tensor,
                        T: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Computes the outputs of the InvariantPointAttention as described
        in the Algorithm 22.

        Args:
            attn_scores:    A PyTorch tensor that includes the invariant
                            attention scores with a shape of
                            (*, N_head, N_res, N_res).
            z:  A PyTorch tensor that includes the pairwise representation
                with a shape of (*, N_res, N_res, c_z).
            v:  A PyTorch tensor that includes the values for the invariant
                point attention with a shape of (*, N_head, N_res, c).
            vp: A PyTorch tensor that includes the point values for the
                invariant point attention  with a shape of
                (*, N_head, n_point_values, N_res, 3).
            T:  A PyTorch tensor that includes the backbone transforms
                with a shape of (*, N_res, 4, 4).

        Returns:
            A tuple of PyTorch tensors:
                * output for the values with a shape of (*, N_res, N_head*c)
                * output for the point values with a shape of
                    (*, N_res, N_head*3*n_point_values)
                * norm of the output for the point values with a shape of
                    (*, N_res, N_head*3*n_point_values)
                * output from the pair representation with a shape of
                    (*, N_res, N_head*c_z).
        """

        # (*, N_res, N_head, c_z)
        out_pairwise = torch.einsum("...hij,...ijc->...ihc", att_scores, z)

        # (*, N_res, N_head, c_z) -> (*, N_res, N_head * c_z)
        out_pairwise = out_pairwise.flatten(start_dim=-2)

        # (*, N_res, N_head, c)
        out_values = torch.einsum("...hij,...hjc->...ihc", att_scores, v)

        # (*, N_res, N_head, c) -> (*, N_res, N_head * c)
        out_values = out_values.flatten(start_dim=-2)

        # (*, N_res, 4, 4) -> (*, 1, 1, N_res, 4, 4)
        T_batch = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))
        T_batch_inv = invert_4x4_transform(T_batch)

        # (*, N_head, n_point_values, N_res, 3)
        out_values_points = torch.einsum("...hij,...hpjc->...hpic", att_scores,
                                         warp_3d_point(T_batch, vp))
        out_values_points = warp_3d_point(T_batch_inv, out_values_points)

        # (*, N_head, n_point_values, N_res, 3) -> (*, N_res, 3, N_head, n_point_values)
        out_values_points = torch.einsum('...hpic->...ichp', out_values_points)
        print("out", out_values_points.shape, v.shape)

        out_norm_values_points = torch.linalg.norm(out_values_points,
                                                   dim=-3,
                                                   keepdim=True)

        # (*, N_head, n_point_values, N_res, 3) -> (*, N_res, N_head*3*n_point_values)
        out_values_points = out_values_points.flatten(start_dim=-3)
        out_norm_values_points = out_norm_values_points.flatten(start_dim=-3)

        return out_values, out_values_points, out_norm_values_points, out_pairwise

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the InvariantPointAttention.

        Args:
            s:  A PyTorch tensor that contains the single representation with a
                shape of (*, N_res, c).
            z:  A PyTorch tensor that contains the pairwise representation with
                a shape of (*, N_res, N_res, c_z).
            T:  A PyTorch tensor that includes the backbone transforms
                with a shape of (*, N_res, 4, 4).
        Returns:
            A PyTorch tensor with the same shape as s.
        """

        q, k, v, qp, kp, vp = self.prepare_qkv(s)

        att_scores = self.compute_attention_scores(q, k, qp, kp, z, T)

        outs = self.compute_outputs(att_scores, z, v, vp, T)

        output = torch.cat(outs, dim=-1)

        return self.linear_out(output)
