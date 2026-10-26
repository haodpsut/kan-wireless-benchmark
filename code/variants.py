"""Hai bien the KAN ma phan bien 1 diem 1 doi: FastKAN va Chebyshev-KAN.

Phan bien hoi: ban do che do co tong quat sang cac ban cai KAN khac khong, hay chi
dung cho spline va Fourier? Hai lop duoi day giu DUNG quy uoc cua harness.py:
2 lop, dem tham so qua nparams(), chon be rong qua match_width().

FastKAN (Li 2024): thay B-spline bang co so RBF Gauss tren luoi co dinh + nhanh base.
  tham so moi lop = din*dout*G (RBF) + dout*din (base) + dout (bias)
Chebyshev-KAN: co so da thuc Chebyshev T_k tren tanh(x), bac K.
  tham so moi lop = din*dout*(K+1) + dout
"""
import torch
import torch.nn as nn


class FastKANLayer(nn.Module):
    """Mot lop FastKAN: moi canh la to hop RBF Gauss, cong mot nhanh tuyen tinh base."""
    def __init__(self, din, dout, grid=8, rng=(-2.0, 2.0), denom=None):
        super().__init__()
        self.din, self.dout, self.G = din, dout, grid
        centers = torch.linspace(rng[0], rng[1], grid)
        self.register_buffer("centers", centers)
        self.denom = denom or (rng[1] - rng[0]) / (grid - 1)
        self.rbf_weight = nn.Parameter(0.1 * torch.randn(dout, din, grid) / (din * grid) ** 0.5)
        self.base_weight = nn.Parameter(0.1 * torch.randn(dout, din) / din ** 0.5)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x):                                  # x: [B, din]
        z = (x[..., None] - self.centers) / self.denom      # [B, din, G]
        phi = torch.exp(-z ** 2)
        out = torch.einsum("big,oig->bo", phi, self.rbf_weight)
        return out + torch.nn.functional.silu(x) @ self.base_weight.t() + self.bias


class ChebyKANLayer(nn.Module):
    """Mot lop Chebyshev-KAN: moi canh la da thuc Chebyshev bac K tren tanh(x)."""
    def __init__(self, din, dout, K=6):
        super().__init__()
        self.din, self.dout, self.K = din, dout, K
        self.coef = nn.Parameter(0.1 * torch.randn(dout, din, K + 1) / (din * (K + 1)) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x):
        u = torch.tanh(x)                                   # dua ve [-1,1] cho Chebyshev
        T = [torch.ones_like(u), u]
        for k in range(2, self.K + 1):
            T.append(2 * u * T[-1] - T[-2])
        basis = torch.stack(T, dim=-1)                      # [B, din, K+1]
        return torch.einsum("bik,oik->bo", basis, self.coef) + self.bias


class KANfast(nn.Module):
    def __init__(self, din, h, dout, grid=8):
        super().__init__()
        self.l1 = FastKANLayer(din, h, grid)
        self.l2 = FastKANLayer(h, dout, grid)

    def forward(self, x):
        return self.l2(self.l1(x))


class KANcheby(nn.Module):
    def __init__(self, din, h, dout, K=6):
        super().__init__()
        self.l1 = ChebyKANLayer(din, h, K)
        self.l2 = ChebyKANLayer(h, dout, K)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))
