# qbc_bvp.py
import numpy as np

def solve_tridiagonal(lower, diag, upper, rhs):
    n = len(diag)
    a = lower.copy()
    b = diag.copy()
    c = upper.copy()
    d = rhs.copy()

    for i in range(1, n):
        w = a[i-1] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]

    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x


def solve_qbc_Xi(d_um,
                 t_um=1.0,
                 N=6001,
                 m_gap=0.07069,
                 m_in=5.0,
                 D_gap=1.0,
                 D_in=1.0):
    z = np.linspace(-t_um, d_um + t_um, N)
    dz = z[1] - z[0]

    in_gap = (z >= 0) & (z <= d_um)

    m = np.where(in_gap, m_gap, m_in)
    D = np.where(in_gap, D_gap, D_in)
    m2 = m**2
    S = m2

    Dph = 0.5 * (D[:-1] + D[1:])

    lower = np.zeros(N-1)
    diag  = np.zeros(N)
    upper = np.zeros(N-1)
    rhs   = np.zeros(N)

    for i in range(1, N-1):
        aL = Dph[i-1] / dz**2
        aR = Dph[i] / dz**2
        lower[i-1] = -aL
        diag[i] = aL + aR + m2[i]
        upper[i] = -aR
        rhs[i] = S[i]

    diag[0] = diag[-1] = 1.0
    rhs[0] = rhs[-1] = 1.0

    Xi = solve_tridiagonal(lower, diag, upper, rhs)

    Xig = Xi[in_gap]
    mean_def = np.trapz(1 - Xig, z[in_gap]) / d_um
    Xi_min = Xig.min()

    return z, Xi, mean_def, Xi_min
       # gF_bounds.py
import numpy as np
import pandas as pd

def compute_gF_bounds(df,
                      Sf_sqrt=1e-17,
                      tau=10.0):
    F_min = Sf_sqrt / np.sqrt(tau)

    out = df.copy()
    out["F_min_N"] = F_min
    out["gF_max_N"] = F_min / df["mean_def_gap"]
    return out
# run_qbc_pipeline.py
import pandas as pd
from qbc_bvp import solve_qbc_Xi

rows = []
for d in range(1, 41):
    _, _, def_gap, Xi_min = solve_qbc_Xi(d)
    rows.append({
        "d_um": d,
        "mean_def_gap": def_gap,
        "Xi_min_gap": Xi_min
    })

df = pd.DataFrame(rows)
df.to_csv("qbc_layered_scan_1to40um.csv", index=False)
print("Saved qbc_layered_scan_1to40um.csv")
