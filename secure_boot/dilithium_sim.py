"""
secure_boot/dilithium_sim.py

CRYSTALS-Dilithium2 secure boot protocol simulation.
Demonstrates the post-quantum firmware verification process described in Section 3.2.

Desktop simulation — models the signing/verification protocol and Merkle tree
construction described in the paper. Timing on desktop will differ from the MCU
values in Table 2; see notes below for embedded deployment.

Paper benchmarks (Table 2, NXP S32K144, Cortex-M4F, 80 MHz):
  SHA3-256 firmware hashing    :  14.2 ms
  Dilithium2 signature verify  :  31.5 ms
  Total boot validation        :  45.7 ms

For actual cryptographic security on embedded hardware use the PQClean C library:
  https://github.com/PQClean/PQClean/tree/master/crypto_sign/dilithium2

NOTE: This script is a protocol-level demonstration. The signing primitive
below uses SHA3/HMAC to model the computational profile of Dilithium.
It does NOT provide real lattice-based post-quantum security and must NOT
be used in a production system.
"""

import hashlib
import hmac
import os
import time
import numpy as np

# ── Dilithium2 parameter reference (NIST Round 3 spec) ────────────────────────
DILITHIUM2 = {
    "n":           256,      # Polynomial ring degree
    "q":       8380417,      # Prime modulus
    "k":             4,      # Module-LWE dimension (rows)
    "l":             4,      # Module-LWE dimension (cols)
    "eta":           2,      # Secret key coefficient range
    "tau":          39,      # Challenge polynomial weight
    "gamma1":  2 ** 17,      # Masking coefficient range
    "security":    128,      # Classical security bits
    "pk_bytes":   1312,      # Public key size
    "sk_bytes":   2528,      # Secret key size
    "sig_bytes":  2420,      # Signature size
}

BLOCK_SIZE = 2048   # Firmware block size for Merkle tree (Section 3.2.2)


# ── Hashing helpers ────────────────────────────────────────────────────────────

def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def merkle_root(firmware: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """
    Build a Merkle hash tree over firmware blocks and return the root hash.
    Implements Equation (1) from Section 3.2.2:
        root = H( H(M1) || H(M2) || ... || H(Mn) )

    Enables segment-level validation without holding the full image in RAM
    during partial OTA updates.
    """
    blocks = [firmware[i: i + block_size]
              for i in range(0, len(firmware), block_size)]
    if not blocks:
        return b"\x00" * 32

    leaves = [sha3_256(b) for b in blocks]

    # Pad leaf list to next power of two for a complete binary tree
    n = 1
    while n < len(leaves):
        n <<= 1
    leaves += [b"\x00" * 32] * (n - len(leaves))

    level = leaves
    while len(level) > 1:
        level = [sha3_256(level[i] + level[i + 1])
                 for i in range(0, len(level), 2)]
    return level[0]


# ── Key generation ─────────────────────────────────────────────────────────────

class DilithiumKeypair:
    """
    Simulated Dilithium2 key pair with correct byte sizes per the specification.

    Real key generation produces:
      Public key  (A, t) : 1312 bytes
      Secret key  (s1,s2): 2528 bytes
    """

    def __init__(self, seed: bytes = None):
        self.seed = seed or os.urandom(32)
        rng = np.random.RandomState(
            int.from_bytes(self.seed[:4], "big") % (2 ** 31)
        )
        self.public_key  = bytes(rng.randint(0, 256, DILITHIUM2["pk_bytes"],
                                              dtype=np.uint8).tolist())
        self._secret_key = bytes(rng.randint(0, 256, DILITHIUM2["sk_bytes"],
                                              dtype=np.uint8).tolist())

    def sign(self, message_hash: bytes) -> bytes:
        """
        Sign a message hash. Returns a simulated signature of the correct
        byte size (2420 bytes).

        Protocol (Section 3.2.1):
          1. Sample y in Z_q^l
          2. w = A * y
          3. c = H(w, M)
          4. z = y + s1 * c   →   output (z, c)
        """
        sig_seed = hmac.new(
            self._secret_key[:32], message_hash, hashlib.sha3_256
        ).digest()
        rng = np.random.RandomState(
            int.from_bytes(sig_seed[:4], "big") % (2 ** 31)
        )
        return bytes(rng.randint(0, 256, DILITHIUM2["sig_bytes"],
                                 dtype=np.uint8).tolist())

    def verify(self, message_hash: bytes, signature: bytes) -> bool:
        """
        Verify a signature.

        Protocol (Section 3.2.1):
          w' = A*z - c*t
          accept if c == H(M, w')
        """
        expected = self.sign(message_hash)
        return hmac.compare_digest(expected, signature)


# ── Manufacturer: sign firmware (done once, offline) ──────────────────────────

def manufacturer_sign(firmware: bytes, keypair: DilithiumKeypair) -> bytes:
    """
    Compute Merkle root of the firmware image and sign it with the
    manufacturer's Dilithium2 private key.
    The resulting signature is stored in the ECU's secure storage.
    """
    root = merkle_root(firmware, BLOCK_SIZE)
    return keypair.sign(root)


# ── ECU: boot-time verification ───────────────────────────────────────────────

def boot_verify(firmware: bytes,
                signature: bytes,
                keypair: DilithiumKeypair) -> tuple:
    """
    Boot-time firmware verification (Section 3.5, Initialization Phase).

    Steps:
      1. SHA3-256 hash the firmware (via Merkle tree)
      2. Verify Dilithium2 signature using the stored public key
      3. Halt boot if signature is invalid

    Returns (is_valid: bool, timing: dict)
    """
    # Phase 1: Hashing
    t0 = time.perf_counter()
    computed_root = merkle_root(firmware, BLOCK_SIZE)
    t1 = time.perf_counter()
    hash_ms = (t1 - t0) * 1000

    # Phase 2: Signature verification
    t2 = time.perf_counter()
    valid = keypair.verify(computed_root, signature)
    t3 = time.perf_counter()
    verify_ms = (t3 - t2) * 1000

    return valid, {
        "hash_ms":   hash_ms,
        "verify_ms": verify_ms,
        "total_ms":  hash_ms + verify_ms,
    }


# ── Report helper ──────────────────────────────────────────────────────────────

def _print_report(firmware: bytes, timing: dict, valid: bool):
    w = 34
    print("=" * 58)
    print("  Secure Boot Simulation — CRYSTALS-Dilithium2")
    print("=" * 58)
    print(f"  Firmware size    : {len(firmware):,} bytes")
    print(f"  PQ security      : {DILITHIUM2['security']}-bit")
    print(f"  Public key size  : {DILITHIUM2['pk_bytes']} bytes")
    print(f"  Signature size   : {DILITHIUM2['sig_bytes']} bytes")
    print()
    print(f"  {'Operation':<{w}} {'This run':>9}   {'Paper (MCU)':>11}")
    print(f"  {'─'*56}")
    print(f"  {'SHA3-256 Firmware Hashing':<{w}} "
          f"{timing['hash_ms']:>8.3f}ms   {'14.2 ms':>11}")
    print(f"  {'Dilithium2 Signature Verify':<{w}} "
          f"{timing['verify_ms']:>8.3f}ms   {'31.5 ms':>11}")
    print(f"  {'─'*56}")
    print(f"  {'Total Boot Validation':<{w}} "
          f"{timing['total_ms']:>8.3f}ms   {'45.7 ms':>11}")
    print()
    icon   = "✓" if valid else "✗"
    status = ("BOOT ACCEPTED — firmware integrity confirmed"
              if valid else
              "BOOT REJECTED — invalid signature, halting ECU")
    print(f"  {icon} {status}")
    print()
    print("  Desktop timing is faster than MCU values (desktop has higher")
    print("  clock speed and no Flash wait states).")
    print("  Paper values: NXP S32K144, Cortex-M4F, 80 MHz, ARM SIMD.")
    print("=" * 58)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Root of trust: manufacturer key pair (seed stored in HSM in production)
    ROOT_SEED = b"automotive_ecu_dilithium2_root_of_trust_v1"
    keypair   = DilithiumKeypair(seed=ROOT_SEED)

    print(f"\nDilithium2 key pair generated")
    print(f"  Public key : {DILITHIUM2['pk_bytes']} bytes")
    print(f"  Secret key : {DILITHIUM2['sk_bytes']} bytes  (never leaves HSM)")

    # ── Test 1: Normal boot — 2048-byte firmware (Table 2 scenario) ──────────
    print("\n" + "─" * 58)
    print("[Test 1] Normal boot — 2048-byte firmware image")
    print("─" * 58)
    fw        = os.urandom(2048)
    signature = manufacturer_sign(fw, keypair)   # Done once, offline
    valid, timing = boot_verify(fw, signature, keypair)
    _print_report(fw, timing, valid)

    # ── Test 2: Tampered firmware — one byte flipped ──────────────────────────
    print("\n" + "─" * 58)
    print("[Test 2] Tampered firmware — original signature, flipped byte 512")
    print("─" * 58)
    fw_tampered       = bytearray(fw)
    fw_tampered[512] ^= 0xFF                           # Simulate injection
    valid_t, timing_t = boot_verify(bytes(fw_tampered), signature, keypair)
    _print_report(bytes(fw_tampered), timing_t, valid_t)
    if not valid_t:
        print("✓ Tamper correctly detected — ECU would halt.\n")

    # ── Test 3: Larger realistic firmware image ───────────────────────────────
    print("\n" + "─" * 58)
    print("[Test 3] Realistic firmware — 64 KB image, multi-block Merkle tree")
    print("─" * 58)
    fw_large        = os.urandom(64 * 1024)
    sig_large       = manufacturer_sign(fw_large, keypair)
    valid_l, t_l    = boot_verify(fw_large, sig_large, keypair)
    n_blocks        = (len(fw_large) + BLOCK_SIZE - 1) // BLOCK_SIZE
    print(f"  Firmware blocks: {n_blocks}  (each {BLOCK_SIZE} bytes)")
    _print_report(fw_large, t_l, valid_l)

    # ── Dilithium2 parameter summary ──────────────────────────────────────────
    print("\nDilithium2 parameter reference (NIST Round 3):")
    for k, v in DILITHIUM2.items():
        print(f"  {k:<12}: {v}")
