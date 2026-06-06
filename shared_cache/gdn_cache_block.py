"""
Gated DeltaNet cache block utilities.

This module implements a small standalone cache block for recurrent GDN-style
linear attention states and their affine summaries.

For a fixed trajectory of GDN token parameters, the recurrent update can be
written as

    S_out = S_in @ A_hat + B_hat

where per-token updates have the form

    S_t = S_{t-1} A_t + B_t
    A_t = alpha_t I - beta_t k_t k_t^T
    B_t = beta_t v_t k_t^T

This file only handles the algebraic cache object. It does not patch HuggingFace
or Qwen internals. The intended integration point is to call
`append_token_update(...)` after a GDN layer has computed k, v, alpha, beta.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


Tensor = torch.Tensor


@dataclass
class GDNCacheBlock:
    """Cache for one contiguous Gated DeltaNet segment.

    Attributes:
        recurrent_state:
            Final recurrent state after the block. Expected shape is
            ``[batch, heads, d_v, d_k]``. In common Qwen-style GDN layers,
            ``d_v == d_k == head_dim``.

        conv_state:
            Optional convolutional state used by implementations that include
            short convolution before the recurrent update. Shape is model-
            specific and intentionally opaque here.

        A_hat, B_hat:
            Optional affine summary of the block:

                S_out = S_in @ A_hat + B_hat

            Expected shapes:
                A_hat: ``[batch, heads, d_k, d_k]``
                B_hat: ``[batch, heads, d_v, d_k]``

        start_pos, end_pos:
            Token span covered by this block. These are bookkeeping values only.
    """

    recurrent_state: Tensor
    conv_state: Optional[Tensor] = None
    A_hat: Optional[Tensor] = None
    B_hat: Optional[Tensor] = None
    start_pos: int = 0
    end_pos: int = 0

    def __post_init__(self) -> None:
        _validate_state(self.recurrent_state, name="recurrent_state")

        if self.conv_state is not None and not torch.is_tensor(self.conv_state):
            raise TypeError("conv_state must be a torch.Tensor or None")

        if (self.A_hat is None) ^ (self.B_hat is None):
            raise ValueError("A_hat and B_hat must either both be provided or both be None")

        if self.A_hat is not None and self.B_hat is not None:
            _validate_affine(self.A_hat, self.B_hat, self.recurrent_state)

        if self.end_pos < self.start_pos:
            raise ValueError("end_pos must be >= start_pos")

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos

    @property
    def has_affine(self) -> bool:
        return self.A_hat is not None and self.B_hat is not None

    @property
    def batch_size(self) -> int:
        return self.recurrent_state.shape[0]

    @property
    def num_heads(self) -> int:
        return self.recurrent_state.shape[1]

    @property
    def d_v(self) -> int:
        return self.recurrent_state.shape[-2]

    @property
    def d_k(self) -> int:
        return self.recurrent_state.shape[-1]

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        num_heads: int,
        d_k: int,
        d_v: Optional[int] = None,
        dtype: torch.dtype,
        device: torch.device | str,
        with_affine: bool = True,
        start_pos: int = 0,
    ) -> "GDNCacheBlock":
        """Create an empty zero-state block.

        If ``with_affine=True``, initializes the affine summary to identity:

            A_hat = I, B_hat = 0

        so applying it to a state returns the same state.
        """

        if d_v is None:
            d_v = d_k

        _validate_positive_int(batch_size, "batch_size")
        _validate_positive_int(num_heads, "num_heads")
        _validate_positive_int(d_k, "d_k")
        _validate_positive_int(d_v, "d_v")

        recurrent_state = torch.zeros(
            batch_size,
            num_heads,
            d_v,
            d_k,
            dtype=dtype,
            device=device,
        )

        A_hat: Optional[Tensor]
        B_hat: Optional[Tensor]
        if with_affine:
            A_hat, B_hat = init_gdn_affine(
                batch_size=batch_size,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                dtype=dtype,
                device=device,
            )
        else:
            A_hat = None
            B_hat = None

        return cls(
            recurrent_state=recurrent_state,
            A_hat=A_hat,
            B_hat=B_hat,
            start_pos=start_pos,
            end_pos=start_pos,
        )

    @classmethod
    def from_state(
        cls,
        recurrent_state: Tensor,
        *,
        conv_state: Optional[Tensor] = None,
        with_affine: bool = False,
        start_pos: int = 0,
        end_pos: Optional[int] = None,
    ) -> "GDNCacheBlock":
        """Create a block from an existing recurrent state."""

        _validate_state(recurrent_state, name="recurrent_state")

        A_hat: Optional[Tensor]
        B_hat: Optional[Tensor]
        if with_affine:
            A_hat, B_hat = init_gdn_affine(
                batch_size=recurrent_state.shape[0],
                num_heads=recurrent_state.shape[1],
                d_v=recurrent_state.shape[-2],
                d_k=recurrent_state.shape[-1],
                dtype=recurrent_state.dtype,
                device=recurrent_state.device,
            )
        else:
            A_hat = None
            B_hat = None

        if end_pos is None:
            end_pos = start_pos

        return cls(
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            A_hat=A_hat,
            B_hat=B_hat,
            start_pos=start_pos,
            end_pos=end_pos,
        )

    def clone(self) -> "GDNCacheBlock":
        return GDNCacheBlock(
            recurrent_state=self.recurrent_state.clone(),
            conv_state=None if self.conv_state is None else self.conv_state.clone(),
            A_hat=None if self.A_hat is None else self.A_hat.clone(),
            B_hat=None if self.B_hat is None else self.B_hat.clone(),
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )

    def detach(self) -> "GDNCacheBlock":
        return GDNCacheBlock(
            recurrent_state=self.recurrent_state.detach(),
            conv_state=None if self.conv_state is None else self.conv_state.detach(),
            A_hat=None if self.A_hat is None else self.A_hat.detach(),
            B_hat=None if self.B_hat is None else self.B_hat.detach(),
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )

    def to(self, *args, **kwargs) -> "GDNCacheBlock":
        return GDNCacheBlock(
            recurrent_state=self.recurrent_state.to(*args, **kwargs),
            conv_state=None if self.conv_state is None else self.conv_state.to(*args, **kwargs),
            A_hat=None if self.A_hat is None else self.A_hat.to(*args, **kwargs),
            B_hat=None if self.B_hat is None else self.B_hat.to(*args, **kwargs),
            start_pos=self.start_pos,
            end_pos=self.end_pos,
        )

    def reset_affine(self) -> None:
        """Reset affine summary to identity for this block shape."""

        self.A_hat, self.B_hat = init_gdn_affine(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            d_v=self.d_v,
            d_k=self.d_k,
            dtype=self.recurrent_state.dtype,
            device=self.recurrent_state.device,
        )

    def apply_affine(self, state: Tensor) -> Tensor:
        """Apply the block affine summary to an arbitrary input state.

        Args:
            state: Tensor with shape ``[batch, heads, d_v, d_k]``.

        Returns:
            Tensor with shape ``[batch, heads, d_v, d_k]``.
        """

        if not self.has_affine:
            raise ValueError("Cannot apply affine summary: A_hat/B_hat are missing")

        assert self.A_hat is not None
        assert self.B_hat is not None

        _validate_state(state, name="state")
        if state.shape != self.B_hat.shape:
            raise ValueError(
                f"state shape must match B_hat shape. Got {tuple(state.shape)} vs "
                f"{tuple(self.B_hat.shape)}"
            )

        return apply_gdn_affine(state, self.A_hat, self.B_hat)

    def append_token_update(
        self,
        *,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        update_recurrent_state: bool = True,
    ) -> None:
        """Append one token update to the block.

        This updates the block-level affine summary using the low-rank form of
        ``A_t``. Optionally also updates ``self.recurrent_state`` as if this
        token were applied to the current final state.

        Expected shapes:
            k:     ``[batch, heads, d_k]``
            v:     ``[batch, heads, d_v]``
            alpha: broadcastable to ``[batch, heads]``
            beta:  broadcastable to ``[batch, heads]``
        """

        if not self.has_affine:
            self.reset_affine()

        assert self.A_hat is not None
        assert self.B_hat is not None

        _validate_token_params(
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        self.A_hat, self.B_hat = update_affine_summary(
            A_hat=self.A_hat,
            B_hat=self.B_hat,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
        )

        if update_recurrent_state:
            self.recurrent_state = gdn_recurrent_update(
                state=self.recurrent_state,
                k=k,
                v=v,
                alpha=alpha,
                beta=beta,
            )

        self.end_pos += 1


def init_gdn_affine(
    *,
    batch_size: int,
    num_heads: int,
    d_k: int,
    d_v: Optional[int] = None,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tuple[Tensor, Tensor]:
    """Initialize identity affine summary for a GDN block.

    Returns:
        A_hat: ``[batch, heads, d_k, d_k]`` identity matrices.
        B_hat: ``[batch, heads, d_v, d_k]`` zeros.
    """

    if d_v is None:
        d_v = d_k

    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(num_heads, "num_heads")
    _validate_positive_int(d_k, "d_k")
    _validate_positive_int(d_v, "d_v")

    eye = torch.eye(d_k, dtype=dtype, device=device)
    A_hat = eye.view(1, 1, d_k, d_k).expand(batch_size, num_heads, d_k, d_k).clone()

    B_hat = torch.zeros(
        batch_size,
        num_heads,
        d_v,
        d_k,
        dtype=dtype,
        device=device,
    )

    return A_hat, B_hat


def apply_gdn_affine(state: Tensor, A_hat: Tensor, B_hat: Tensor) -> Tensor:
    """Apply ``S_out = S_in @ A_hat + B_hat``."""

    _validate_state(state, name="state")
    _validate_affine(A_hat, B_hat, state)
    return torch.matmul(state, A_hat) + B_hat


def gdn_recurrent_update(
    *,
    state: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> Tensor:
    """Apply one GDN recurrent update directly to ``state``.

    Uses the low-rank form:

        S_t = alpha * S_{t-1}
              - beta * (S_{t-1} k_t) k_t^T
              + beta * v_t k_t^T

    Expected shapes:
        state: ``[batch, heads, d_v, d_k]``
        k:     ``[batch, heads, d_k]``
        v:     ``[batch, heads, d_v]``
        alpha: broadcastable to ``[batch, heads]``
        beta:  broadcastable to ``[batch, heads]``
    """

    _validate_state(state, name="state")
    _validate_token_params(
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        batch_size=state.shape[0],
        num_heads=state.shape[1],
        d_k=state.shape[-1],
        d_v=state.shape[-2],
    )

    alpha_b = _as_head_scalar(alpha, reference=k)
    beta_b = _as_head_scalar(beta, reference=k)

    state_k = torch.matmul(state, k.unsqueeze(-1)).squeeze(-1)  # [B, H, Dv]

    decayed = alpha_b.unsqueeze(-1) * state
    erased = (alpha_b.unsqueeze(-1) * beta_b.unsqueeze(-1) * state_k.unsqueeze(-1) * k.unsqueeze(-2))
    written = beta_b.unsqueeze(-1) * v.unsqueeze(-1) * k.unsqueeze(-2)

    return decayed - erased + written


def update_affine_summary(
    *,
    A_hat: Tensor,
    B_hat: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Append one GDN token update to an existing affine block summary.

    Existing summary:
        S_mid = S_in @ A_hat + B_hat

    New update:
        S_out = S_mid @ A_t + B_t

    New summary:
        A_hat_new = A_hat @ A_t
        B_hat_new = B_hat @ A_t + B_t

    Since ``A_t = alpha I - beta k k^T``, multiplication by ``A_t`` is
    computed as a rank-1 update instead of dense ``D x D`` matmul.
    """

    _validate_affine_tensors(A_hat, B_hat)
    _validate_token_params(
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        batch_size=B_hat.shape[0],
        num_heads=B_hat.shape[1],
        d_k=B_hat.shape[-1],
        d_v=B_hat.shape[-2],
    )

    alpha_b = _as_head_scalar(alpha, reference=k)
    beta_b = _as_head_scalar(beta, reference=k)

    # A_hat @ A_t = alpha * A_hat - beta * (A_hat @ k) k^T
    A_k = torch.matmul(A_hat, k.unsqueeze(-1)).squeeze(-1)  # [B, H, Dk]
    A_hat_new = alpha_b.unsqueeze(-1) * A_hat - (
        alpha_b.unsqueeze(-1)
        * beta_b.unsqueeze(-1)
        * A_k.unsqueeze(-1)
        * k.unsqueeze(-2)
    )

    # B_hat @ A_t = alpha * B_hat - beta * (B_hat @ k) k^T
    B_k = torch.matmul(B_hat, k.unsqueeze(-1)).squeeze(-1)  # [B, H, Dv]
    B_hat_new = alpha_b.unsqueeze(-1) * B_hat - (
        alpha_b.unsqueeze(-1)
        * beta_b.unsqueeze(-1)
        * B_k.unsqueeze(-1)
        * k.unsqueeze(-2)
    )

    # B_t = beta * v k^T
    B_t = beta_b.unsqueeze(-1) * v.unsqueeze(-1) * k.unsqueeze(-2)
    B_hat_new = B_hat_new + B_t

    return A_hat_new, B_hat_new


def compose_gdn_affines(
    *,
    A_first: Tensor,
    B_first: Tensor,
    A_second: Tensor,
    B_second: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compose two affine GDN summaries.

    First:
        S_mid = S_in @ A_first + B_first

    Second:
        S_out = S_mid @ A_second + B_second

    Composition:
        S_out = S_in @ (A_first @ A_second) + (B_first @ A_second + B_second)
    """

    _validate_affine_tensors(A_first, B_first)
    _validate_affine_tensors(A_second, B_second)

    if A_first.shape != A_second.shape:
        raise ValueError(
            f"A shapes must match. Got {tuple(A_first.shape)} and {tuple(A_second.shape)}"
        )
    if B_first.shape != B_second.shape:
        raise ValueError(
            f"B shapes must match. Got {tuple(B_first.shape)} and {tuple(B_second.shape)}"
        )

    A = torch.matmul(A_first, A_second)
    B = torch.matmul(B_first, A_second) + B_second
    return A, B


def _as_head_scalar(x: Tensor, *, reference: Tensor) -> Tensor:
    """Normalize scalar gates to shape ``[batch, heads, 1]``.

    ``reference`` is usually k with shape ``[batch, heads, d_k]``.
    """

    if not torch.is_tensor(x):
        raise TypeError("gate must be a torch.Tensor")

    batch_size, num_heads = reference.shape[:2]

    if x.ndim == 0:
        x = x.view(1, 1).expand(batch_size, num_heads)
    elif x.ndim == 1:
        if x.shape[0] == batch_size:
            x = x[:, None].expand(batch_size, num_heads)
        elif x.shape[0] == num_heads:
            x = x[None, :].expand(batch_size, num_heads)
        else:
            raise ValueError(
                f"1D gate must have length batch_size={batch_size} or "
                f"num_heads={num_heads}; got {x.shape[0]}"
            )
    elif x.ndim == 2:
        if x.shape != (batch_size, num_heads):
            raise ValueError(
                f"2D gate must have shape {(batch_size, num_heads)}; got {tuple(x.shape)}"
            )
    elif x.ndim == 3:
        if x.shape != (batch_size, num_heads, 1):
            raise ValueError(
                f"3D gate must have shape {(batch_size, num_heads, 1)}; got {tuple(x.shape)}"
            )
        return x
    else:
        raise ValueError(f"gate must have ndim <= 3; got ndim={x.ndim}")

    return x.unsqueeze(-1)


def _validate_state(state: Tensor, *, name: str) -> None:
    if not torch.is_tensor(state):
        raise TypeError(f"{name} must be a torch.Tensor")
    if state.ndim != 4:
        raise ValueError(
            f"{name} must have shape [batch, heads, d_v, d_k]; got {tuple(state.shape)}"
        )
    if any(dim <= 0 for dim in state.shape):
        raise ValueError(f"{name} must have all positive dimensions; got {tuple(state.shape)}")


def _validate_affine(A_hat: Tensor, B_hat: Tensor, state: Tensor) -> None:
    _validate_state(state, name="state")
    _validate_affine_tensors(A_hat, B_hat)

    batch_size, num_heads, d_v, d_k = state.shape
    expected_A = (batch_size, num_heads, d_k, d_k)
    expected_B = (batch_size, num_heads, d_v, d_k)

    if tuple(A_hat.shape) != expected_A:
        raise ValueError(f"A_hat must have shape {expected_A}; got {tuple(A_hat.shape)}")
    if tuple(B_hat.shape) != expected_B:
        raise ValueError(f"B_hat must have shape {expected_B}; got {tuple(B_hat.shape)}")

    if A_hat.device != state.device or B_hat.device != state.device:
        raise ValueError("A_hat, B_hat, and state must be on the same device")
    if A_hat.dtype != state.dtype or B_hat.dtype != state.dtype:
        raise ValueError("A_hat, B_hat, and state must have the same dtype")


def _validate_affine_tensors(A_hat: Tensor, B_hat: Tensor) -> None:
    if not torch.is_tensor(A_hat):
        raise TypeError("A_hat must be a torch.Tensor")
    if not torch.is_tensor(B_hat):
        raise TypeError("B_hat must be a torch.Tensor")
    if A_hat.ndim != 4:
        raise ValueError(f"A_hat must have shape [batch, heads, d_k, d_k]; got {tuple(A_hat.shape)}")
    if B_hat.ndim != 4:
        raise ValueError(f"B_hat must have shape [batch, heads, d_v, d_k]; got {tuple(B_hat.shape)}")
    if A_hat.shape[0] != B_hat.shape[0] or A_hat.shape[1] != B_hat.shape[1]:
        raise ValueError("A_hat and B_hat must have matching batch/head dimensions")
    if A_hat.shape[-1] != A_hat.shape[-2]:
        raise ValueError("A_hat must be square in its last two dimensions")
    if A_hat.shape[-1] != B_hat.shape[-1]:
        raise ValueError("A_hat d_k must match B_hat d_k")
    if A_hat.device != B_hat.device:
        raise ValueError("A_hat and B_hat must be on the same device")
    if A_hat.dtype != B_hat.dtype:
        raise ValueError("A_hat and B_hat must have the same dtype")


def _validate_token_params(
    *,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    batch_size: int,
    num_heads: int,
    d_k: int,
    d_v: int,
) -> None:
    if not torch.is_tensor(k):
        raise TypeError("k must be a torch.Tensor")
    if not torch.is_tensor(v):
        raise TypeError("v must be a torch.Tensor")

    expected_k = (batch_size, num_heads, d_k)
    expected_v = (batch_size, num_heads, d_v)

    if tuple(k.shape) != expected_k:
        raise ValueError(f"k must have shape {expected_k}; got {tuple(k.shape)}")
    if tuple(v.shape) != expected_v:
        raise ValueError(f"v must have shape {expected_v}; got {tuple(v.shape)}")

    if k.device != v.device:
        raise ValueError("k and v must be on the same device")
    if k.dtype != v.dtype:
        raise ValueError("k and v must have the same dtype")

    # Validate gates and make sure they are broadcastable to [B, H, 1].
    _as_head_scalar(alpha, reference=k)
    _as_head_scalar(beta, reference=k)


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int; got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
