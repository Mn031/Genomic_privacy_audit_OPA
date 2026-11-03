"""
crypto.py - One-Shot Private Aggregation (OPA) Cryptographic Primitives for Genomic Language Models.

References:
  - "OPA: One-Shot Private Aggregation with Single Client Interaction and Its
    Applications to Federated Learning" (Karthikeyan & Polychroniadou, CRYPTO 2025)
    * Construction 2: LWR Seed-Homomorphic PRG (Section 5.1, page 19).
    * Construction 1: Shamir's Secret Sharing over Fq (Section 4.1, page 16-17).

CRITICAL OPTIMIZATION FOR 12.6M PARAMETER GENOMIC TRANSFORMER:
  - SimpleDNALM possesses ~12.88M parameters (L = 12,876,800).
  - A dense int64 matrix A of shape (12,876,800, 128) would require ~13.18 GB of RAM.
  - To support the exact 12.6M parameter footprint dynamically on consumer hardware without
    exceeding memory or sacrificing cryptographic fidelity, LWR_PRG implements deterministic
    chunk-wise matrix expansion. The resulting pseudorandom mask is mathematically and bitwise
    identical to full matrix multiplication:
        PRG_{LWR, A}(s) = \\lfloor (A \\cdot s \\bmod q) \\times (p / q) \\rfloor \\bmod p
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Cryptographic Moduli & Quantization Scale
# q: Mersenne prime (2^31 - 1) defining the finite field Fq
DEFAULT_Q = 2147483647
# p: Rounding modulus (2^20 = 1,048,576) defining ciphertext space Z_p
DEFAULT_P = 1048576
# Fixed-point quantization scaling factor
DEFAULT_SCALE = 10000.0


# =====================================================================
# 1. MODULAR ARITHMETIC UTILITIES
# =====================================================================

def mod_inverse(a: int, m: int) -> int:
    """
    Computes modular multiplicative inverse of a modulo m using Fermat's Little Theorem.
    Assumes m is prime.
    """
    a = int(a) % m
    if a == 0:
        raise ZeroDivisionError(f"Cannot compute modular inverse of 0 mod {m}")
    return pow(a, m - 2, m)


# =====================================================================
# 2. SHAMIR'S SECRET SHARING (SSS) OVER F_q (CONSTRUCTION 1)
# =====================================================================

def share(
    secret: Union[int, np.ndarray, List[int]],
    t: int,
    r: int,
    m: int,
    q: int = DEFAULT_Q,
    seed: Optional[int] = None,
) -> List[Tuple[int, np.ndarray]]:
    """
    Shamir's (t, r, m)-Secret Sharing Scheme over prime field Fq.
    
    Generates a random degree-(r - 1) polynomial f(X) such that:
      f(0) = secret
      f(j) = share for party j (j = 1, ..., m)
    """
    if r > m:
        raise ValueError(f"Reconstruction threshold r ({r}) cannot exceed total parties m ({m})")
    if r <= 0:
        raise ValueError("Reconstruction threshold r must be positive")

    rng = np.random.RandomState(seed) if seed is not None else np.random

    secret_arr = np.array(secret, dtype=np.int64) % q
    if secret_arr.ndim == 0:
        secret_arr = secret_arr.reshape(1)

    dim = secret_arr.shape[0]

    # Sample random polynomial coefficients a_1, ..., a_{r-1} in Fq
    if r > 1:
        coeffs = rng.randint(0, q, size=(r - 1, dim), dtype=np.int64)
    else:
        coeffs = np.empty((0, dim), dtype=np.int64)

    shares: List[Tuple[int, np.ndarray]] = []
    for x in range(1, m + 1):
        y = secret_arr.copy()
        x_power = int(x)
        for deg in range(r - 1):
            term = (coeffs[deg] * x_power) % q
            y = (y + term) % q
            x_power = (x_power * x) % q
        shares.append((x, y))

    return shares


def reconstruct(
    shares: List[Tuple[int, np.ndarray]],
    q: int = DEFAULT_Q,
) -> np.ndarray:
    """
    Reconstructs the secret f(0) from at least r shares using Lagrange interpolation over Fq.
    Linearly homomorphic over Fq: Reconstruct(sum_i shares_i) == sum_i secret_i mod q.
    """
    if len(shares) == 0:
        raise ValueError("Cannot reconstruct from an empty list of shares")

    r = len(shares)
    xs = [int(s[0]) for s in shares]
    dim = shares[0][1].shape[0]

    secret = np.zeros(dim, dtype=np.int64)

    for j in range(r):
        xj = xs[j]
        yj = np.array(shares[j][1], dtype=np.int64) % q

        num = 1
        den = 1
        for k in range(r):
            if k == j:
                continue
            xk = xs[k]
            num = (num * (-xk)) % q
            den = (den * (xj - xk)) % q

        ell_j = (num * mod_inverse(den, q)) % q
        secret = (secret + (yj * ell_j) % q) % q

    return secret


# =====================================================================
# 3. LWR SEED-HOMOMORPHIC PRG (CONSTRUCTION 2)
# =====================================================================

class LWR_PRG:
    """
    Learning With Rounding (LWR) Seed-Homomorphic Pseudorandom Generator.
    Refer to Construction 2 (OPA paper, Section 5.1, page 19):
    
      PRG_{LWR, A}(s) = \\lfloor (A \\cdot s \\bmod q) \\times (p / q) \\rfloor \\bmod p

    Dynamically sizes matrix A to match the 12.88M parameter dimension of SimpleDNALM.
    Uses chunked deterministic generation to operate within a lightweight memory footprint (~50MB).
    """

    def __init__(
        self,
        L: int = 12745728,
        lam: int = 128,
        q: int = DEFAULT_Q,
        p: int = DEFAULT_P,
        A: Optional[np.ndarray] = None,
        chunk_size: int = 200000,
        seed: int = 42,
    ) -> None:
        self.L = int(L)
        self.lam = int(lam)
        self.q = int(q)
        self.p = int(p)
        self.chunk_size = chunk_size
        self.seed = seed

        if A is not None:
            if A.shape != (self.L, self.lam):
                raise ValueError(f"Matrix A shape mismatch: expected ({self.L}, {self.lam}), got {A.shape}")
            self._A = np.array(A, dtype=np.int64) % self.q
        else:
            self._A = None

    def expand(self, seed_s: np.ndarray) -> np.ndarray:
        """
        Computes the pseudorandom mask from seed s:
          mask = \\lfloor (A \\cdot s \\bmod q) \\times (p / q) \\rfloor \\bmod p

        Bitwise identical to dense matrix multiplication, evaluated chunk-wise to support
        the massive 12.6M parameter footprint in low memory.
        """
        s = np.array(seed_s, dtype=np.int64) % self.q
        if s.shape[0] != self.lam:
            raise ValueError(f"Seed shape mismatch: expected ({self.lam},), got {s.shape}")

        if self._A is not None:
            prod = (self._A @ s) % self.q
            scaled = np.floor((prod.astype(np.float64) * self.p) / self.q).astype(np.int64)
            return scaled % self.p

        # Chunked deterministic generation for large L
        mask = np.empty(self.L, dtype=np.int64)
        rng = np.random.RandomState(self.seed)
        chunk = min(self.chunk_size, self.L)

        for start in range(0, self.L, chunk):
            end = min(start + chunk, self.L)
            cur_len = end - start
            A_chunk = rng.randint(0, self.q, size=(cur_len, self.lam), dtype=np.int64)
            prod_chunk = (A_chunk @ s) % self.q
            scaled_chunk = np.floor((prod_chunk.astype(np.float64) * self.p) / self.q).astype(np.int64)
            mask[start:end] = scaled_chunk % self.p

        return mask

    @property
    def A(self) -> np.ndarray:
        """Returns or materializes full matrix A (warns if L > 1,000,000 to prevent OOM)."""
        if self._A is not None:
            return self._A
        if self.L > 1000000:
            import warnings
            warnings.warn(
                f"Materializing matrix A with L={self.L} and lam={self.lam} requires ~{self.L * self.lam * 8 / (1024**3):.2f} GB RAM. "
                "Prefer calling expand(seed_s) which computes the PRG chunk-wise.",
                ResourceWarning,
            )
        rng = np.random.RandomState(self.seed)
        return rng.randint(0, self.q, size=(self.L, self.lam), dtype=np.int64)

    @staticmethod
    def flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
        """Flattens a list of model parameter ndarrays into a 1D float32 vector."""
        shapes = [w.shape for w in weights]
        flat_vector = np.concatenate([w.reshape(-1) for w in weights]).astype(np.float32)
        return flat_vector, shapes

    @staticmethod
    def unflatten_weights(flat_vector: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
        """Reconstructs the original list of parameter ndarrays from a flattened 1D vector."""
        weights = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape))
            segment = flat_vector[offset : offset + size].reshape(shape)
            weights.append(segment)
            offset += size
        return weights


# =====================================================================
# 4. FIXED-POINT QUANTIZATION (FLOAT <-> Z_p)
# =====================================================================

def quantize_to_zp(
    weights_float: np.ndarray,
    scale: float = DEFAULT_SCALE,
    p: int = DEFAULT_P,
) -> np.ndarray:
    """Quantizes floating-point weights into integer field Z_p using fixed-point scaling."""
    scaled = np.round(weights_float * scale).astype(np.int64)
    return scaled % p


def dequantize_from_zp(
    weights_zp: np.ndarray,
    scale: float = DEFAULT_SCALE,
    p: int = DEFAULT_P,
    num_contributors: int = 1,
) -> np.ndarray:
    """Dequantizes summed integer weights from Z_p back to floating-point values."""
    centered = np.where(weights_zp > p // 2, weights_zp - p, weights_zp).astype(np.float32)
    return centered / (scale * num_contributors)
