import math
import torch


class MultiHeadAttention(torch.nn.Module):
    """
    MultiHeadAttention module.
    """

    def __init__(self,
                 c_in: int,
                 c: int,
                 N_head: int,
                 attn_dim: int,
                 gated=False,
                 is_global=False,
                 use_bias_for_embeddings=False) -> None:
        """
        Initializes the module.

        Args:
            c_in:   Input dimension for the embeddings
            c:  Embedding dimension for each head.
            N_head: Number of heads.
            attn_dim:   The dimension of the input tensor in which the attention
                will be performed.
            gated:  If True, use sigmoid activated gated attention mechanism.
            is_global:  If True, use global attention mechanism by averaging
                the query vectors to one query vector.
            use_bias_for_embeddings:    If True, use bias for query, key, and
                value vectors.
        """

        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.attn_dim = attn_dim
        self.gated = gated
        self.is_global = is_global
        self.use_bias_for_embeddings = use_bias_for_embeddings

        self.linear_q = torch.nn.Linear(c_in,
                                        c * N_head,
                                        bias=use_bias_for_embeddings)

        c_kv = c if is_global else c * N_head
        self.linear_k = torch.nn.Linear(c_in,
                                        c_kv,
                                        bias=use_bias_for_embeddings)
        self.linear_v = torch.nn.Linear(c_in,
                                        c_kv,
                                        bias=use_bias_for_embeddings)

        self.linear_o = torch.nn.Linear(c * N_head, c_in)

        if gated:
            self.linear_g = torch.nn.Linear(c_in, c * N_head)

    def prepare_qkv(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the embeddings into individual heads by transforming tensors
        from the shape (*, q/k/v, *, N_head*c) to the shape
        (*, N_head, q/k/v, c) where the position of q/k/v is self.attn_dim.

        Args:
            q:  A PyTorch tensor for the query embeddings in the shape of
                (*, q, *, N_head*c)
            k:  A PyTorch tensor for the key embeddings in the shape of
                (*, k, *, N_head*c)
            v:  A PyTorch tensor for the value embeddings in the shape of
                (*, v, *, N_head*c)

        Returns:
            The rearranged q,k, and v tensors in the shape of
            (*, N_head, q/k/v, c).
        """

        # (*, q/k/v, *, N_head*c) -> (*, q/k/v, N_head*c)
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        # (*, q/k/v, N_head*c) -> (*, q/k/v, N_head, c)
        q = q.view(q.shape[:-1] + (self.N_head, -1))
        k = k.view(k.shape[:-1] + (self.N_head, -1))
        v = v.view(v.shape[:-1] + (self.N_head, -1))

        # (*, q/k/v, N_head, c) -> (*, N_head, q/k/v, c)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        return q, k, v

    def prepare_qkv_global(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares q,k, and v PyTorch tensors for global attention. Differences
        from the prepare_qkv:
            - One head for the key and value embeddings
            - The query vectors are averaged to one vector

        Args:
            q:  A PyTorch tensor for the query embeddings in the shape of
                (*, q, *, N_head*c)
            k:  A PyTorch tensor for the key embeddings in the shape of
                (*, k, *, c)
            v:  A PyTorch tensor for the value embeddings in the shape of
                (*, v, *, c)

        Returns:
            The rearranged q,k, and v tensors in the shape of
            (*, N_head, q/k/v, c) for q and (*, 1, k, c) for k and v.
        """

        # (*, q/k/v, *, N_head*c) -> (*, q/k/v, N_head*c)
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        # (*, q/k/v, N_head*c) -> (*, q/k/v, N_head, c)
        q = q.view(q.shape[:-1] + (self.N_head, -1))

        # (*, q/k/v, N_head, c) -> (*, N_head, q/k/v, c)
        q = q.transpose(-2, -3)
        q = q.mean(dim=-2, keepdim=True)

        # (*, q/k/v, 1, c) -> (*, 1, q/k/v, c)
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        return q, k, v

    def forward(self,
                x: torch.Tensor,
                bias: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the MultiHeadAttention module.

        Args:
            x:  A PyTorch tensor of shape (*, q/k/v, *, c_in) that contains
                the input.
            bias:   A PyTorch tensor of shape (*, N_head, q, k) that should
                be added to the attention weights.
            attention_mask: A PyTorch tensor of shape (*, k) that contains
                a mask where the values that should not be used are marked with
                0.

        Returns:
            Output PyTorch tensor with a shape of (*, q/k/v, *, c_in).
        """

        # (*, q/k/v, *, N_head*c)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # (*, q/k/v, *, N_head*c) -> (*, N_head, q/k/v, c)
        if self.is_global:
            q, k, v = self.prepare_qkv_global(q, k, v)
        else:
            q, k, v = self.prepare_qkv(q, k, v)

        q = q / math.sqrt(self.c)

        a = torch.einsum("...qc,...kc->...qk", q, k)
        if bias is not None:
            bias_batch_shape = bias.shape[:-3]
            bias_bc_shape = bias_batch_shape + (1, ) * (
                a.ndim - len(bias_batch_shape) - 3) + bias.shape[-3:]

            a = a + bias.view(bias_bc_shape)

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            # offset = (attention_mask==0) * -1e8
            # a = a + offset
            a[~attention_mask] = -torch.inf

        a = torch.softmax(a, dim=-1)

        # (*, N_head, q, c)
        o = torch.einsum("...qk,...kc->...qc", a, v)

        # (*, N_head, q, c) -> (*, q, N_head, c)
        o = o.transpose(-3, -2)
        # (*, q, N_head, c) -> (*, q, N_head*c)
        o = o.flatten(start_dim=-2)
        # (*, q, N_head*c) -> (*, q, *, N_head * c)
        o = o.movedim(-2, self.attn_dim)

        if self.gated:
            g = torch.sigmoid(self.linear_g(x))
            o = g * o

        o = self.linear_o(o)
        return o
