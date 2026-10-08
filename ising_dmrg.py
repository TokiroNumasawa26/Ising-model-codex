"""Open transverse-field Ising chain, Pauli convention, entropy in nats."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from scipy.sparse.linalg import eigsh


def hamiltonian(n, j, h):
    builder = qtn.SpinHam1D(S=0.5, cyclic=False)
    # SpinHam uses S=Pauli/2, hence factors of four and two.
    builder += -4 * j, "Z", "Z"
    builder += -2 * h, "X"
    return builder.build_mpo(n)


def ground_state(n, j, h, bonds=(16, 32, 64, 128), tol=1e-10):
    qu.seed_rand(2026)
    initial = qtn.MPS_rand_state(n, bond_dim=8, dtype="float64")
    solver = qtn.DMRG2(hamiltonian(n, j, h), bond_dims=bonds,
                       cutoffs=1e-12, p0=initial)
    converged = solver.solve(tol=tol, max_sweeps=20, verbosity=1)
    if not converged:
        raise RuntimeError("DMRG energy did not converge in 20 sweeps")
    state = solver.state.copy()
    state.normalize()
    return state, solver


def observables(state):
    n = state.L
    z = qu.pauli("Z").real
    entropy = np.array([state.entropy(i) * np.log(2) for i in range(1, n)])
    mz = np.array([float(np.real(state.local_expectation_exact(z, (i,))))
                   for i in range(n)])
    zz = np.eye(n)
    for i in range(n):
        for k in range(i + 1, n):
            zz[i, k] = zz[k, i] = float(np.real(
                state.local_expectation_exact(np.kron(z, z), (i, k))))
    return entropy, mz, zz, zz - np.outer(mz, mz)


def validate():
    """Independent 8-site sparse exact diagonalization comparison."""
    n = 8
    state, solver = ground_state(n, 1.0, 1.0)
    dim = 2 ** n
    basis = np.arange(dim)
    signs = np.array([1 - 2 * ((basis >> (n - 1 - i)) & 1)
                      for i in range(n)])
    from scipy.sparse import diags, coo_matrix
    ham = diags(-np.sum(signs[:-1] * signs[1:], axis=0), dtype=float)
    for i in range(n):
        ham += coo_matrix((-np.ones(dim), (basis, basis ^ (1 << i))),
                          shape=(dim, dim)).tocsr()
    energy, vec = eigsh(ham, k=1, which="SA", v0=np.ones(dim), tol=1e-13)
    vec = vec[:, 0]
    entropy, mz, zz, _ = observables(state)
    exact_entropy = []
    for cut in range(1, n):
        p = np.linalg.svd(vec.reshape(2 ** cut, -1), compute_uv=False) ** 2
        p = p[p > 1e-15]
        exact_entropy.append(-np.sum(p * np.log(p)))
    exact_zz = (signs * (vec ** 2)) @ signs.T
    errors = {"energy_abs_error": float(abs(solver.energy - energy[0])),
              "entropy_max_abs_error": float(np.max(abs(entropy - exact_entropy))),
              "zz_max_abs_error": float(np.max(abs(zz - exact_zz)))}
    assert errors["energy_abs_error"] < 1e-8, errors
    assert errors["entropy_max_abs_error"] < 1e-6, errors
    assert errors["zz_max_abs_error"] < 1e-6, errors
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.sites < 2:
        parser.error("--sites must be at least 2")
    checks = validate() if args.validate else None
    state, solver = ground_state(args.sites, args.J, args.h)
    entropy, mz, zz, connected = observables(state)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "entropy.csv", np.column_stack((np.arange(1, args.sites), entropy)),
               delimiter=",", header="cut,entropy_nats", comments="")
    np.savetxt(out / "zz_matrix.csv", zz, delimiter=",")
    np.savetxt(out / "connected_zz_matrix.csv", connected, delimiter=",")
    np.savetxt(out / "magnetization_z.csv", np.column_stack((np.arange(args.sites), mz)),
               delimiter=",", header="site_0_based,mean_Z", comments="")
    reference = args.sites // 2 - 1
    targets = np.arange(reference + 1, args.sites)
    distance = targets - reference
    np.savetxt(out / "correlation.csv", np.column_stack((distance, zz[reference, targets],
               connected[reference, targets])), delimiter=",",
               header="distance,ZZ,connected_ZZ", comments="")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(np.arange(1, args.sites), entropy, "o-", ms=4)
    axes[0].set(xlabel="Cut position l", ylabel="S(l) [nats]",
                title="Bipartite entanglement entropy")
    axes[1].plot(distance, zz[reference, targets], "o-", label=r"$\langle Z_i Z_{i+r}\rangle$")
    axes[1].plot(distance, connected[reference, targets], "x--", label="Connected")
    axes[1].set(xlabel="Distance r", ylabel="Correlation",
                title=f"Reference site i={reference} (0-based)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Open Ising chain: N={args.sites}, J={args.J:g}, h={args.h:g}")
    fig.savefig(out / "ising_dmrg.png", dpi=180)
    fig.savefig(out / "ising_dmrg.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    plot = ax.imshow(zz, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set(xlabel="Site j (0-based)", ylabel="Site i (0-based)", title="ZZ correlation matrix")
    fig.colorbar(plot, ax=ax)
    fig.savefig(out / "correlation_matrix.png", dpi=180)
    plt.close(fig)
    summary = {"sites": args.sites, "J": args.J, "h": args.h, "boundary": "open",
               "hamiltonian": "-J sum Z_i Z_(i+1) - h sum X_i (Pauli)",
               "energy": float(np.real(solver.energy)), "converged": True,
               "sweep_energies": [float(np.real(e)) for e in solver.energies],
               "max_bond": state.max_bond(), "entropy_units": "nats",
               "half_chain_entropy": float(entropy[args.sites // 2 - 1]),
               "max_abs_mean_Z": float(np.max(abs(mz))),
               "validation_8_sites": checks}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
