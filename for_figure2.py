"""
Python code for power series solutions to the SIR-type model.
The initial version of this code was written using the assistance of Claude (https://claude.ai).

Comparison of 9th-order Power Series Solutions vs High-Accuracy ODE45 (RK45)
for the SIR-type system:
    S'(t) = -b * S(t) * I(t)
    I'(t) =  b * S(t) * I(t) - c * I(t)

Parameters:  b = 0.5,  c = 0.25,  S0 = 0.99,  I0 = 0.01
Time interval: [0, tend]. Here tend is the time end.
"""

from fractions import Fraction
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Exact rational power-series coefficients  (order 0 … 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_power_series_coeffs(b_frac, c_frac, S0_frac, I0_frac, order=9):
    """
    Return lists s[0..order], i[0..order] as Python Fraction objects using
    the discrete-convolution recurrence:
        s[n+1] = -b/(n+1) * sum_{k=0}^{n} s[k]*i[n-k]
        i[n+1] = (b * conv_n - c * i[n]) / (n+1)
    """
    N = order + 1
    s = [Fraction(0)] * N
    iv = [Fraction(0)] * N
    s[0]  = S0_frac
    iv[0] = I0_frac

    for n in range(N - 1):
        conv = sum(s[k] * iv[n - k] for k in range(n + 1))
        s[n + 1]  = -b_frac * conv / (n + 1)
        iv[n + 1] = (b_frac * conv - c_frac * iv[n]) / (n + 1)

    return s, iv


b_frac  = Fraction(1, 2)
c_frac  = Fraction(1, 4)
S0_frac = Fraction(99, 100)
I0_frac = Fraction(1, 100)

s_coeffs, i_coeffs = compute_power_series_coeffs(b_frac, c_frac, S0_frac, I0_frac)

# Convert to float numpy arrays for fast polynomial evaluation
s_float = np.array([float(c) for c in s_coeffs])   # s[0] + s[1]*t + ... + s[9]*t^9
i_float = np.array([float(c) for c in i_coeffs])

print("Power-series coefficients (exact fractions):")
print(f"{'n':>3}  {'s_n':>40}  {'i_n':>40}")
print("-" * 87)
for n in range(10):
    print(f"{n:>3}  {str(s_coeffs[n]):>40}  {str(i_coeffs[n]):>40}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Power-series evaluation   S_ps(t) = sum_{n=0}^{9} s_n * t^n
# ─────────────────────────────────────────────────────────────────────────────

def eval_power_series(t_arr, coeffs):
    """Horner's method for sum_{n=0}^{N-1} coeffs[n] * t^n."""
    result = np.zeros_like(t_arr, dtype=float)
    for c in reversed(coeffs):
        result = result * t_arr + float(c)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 3.  High-accuracy ODE45 (RK45) reference solution
# ─────────────────────────────────────────────────────────────────────────────

b_num = 0.5
c_num = 0.25
S0    = 0.99
I0    = 0.01

def sir_rhs(t, y):
    S, I = y
    dS = -b_num * S * I
    dI =  b_num * S * I - c_num * I
    return [dS, dI]

tend = 20.0
t_span = (0.0, tend)
t_eval = np.linspace(0.0, tend, 5000)

sol = solve_ivp(
    sir_rhs,
    t_span,
    [S0, I0],
    method="RK45",
    t_eval=t_eval,
    rtol=1e-12,
    atol=1e-14,
    dense_output=True,
)

print(f"\nODE45 solver status  : {sol.status}  ({sol.message})")
print(f"Number of RHS evaluations : {sol.nfev}")

S_num = sol.y[0]   # high-accuracy numerical S(t)
I_num = sol.y[1]   # high-accuracy numerical I(t)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Power-series approximation on the same t grid
# ─────────────────────────────────────────────────────────────────────────────

S_ps = eval_power_series(t_eval, s_float)
I_ps = eval_power_series(t_eval, i_float)

# Absolute errors
err_S = np.abs(S_ps - S_num)
err_I = np.abs(I_ps - I_num)
err_R = np.abs((1-S_ps-I_ps) - (1-S_num-I_num))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Print a table of values at selected time points
# ─────────────────────────────────────────────────────────────────────────────

t_check = np.array([1.0, 2.0, 4.0, 8.0])
S_num_c = sol.sol(t_check)[0]
I_num_c = sol.sol(t_check)[1]
R_num_c = 1 - S_num_c - I_num_c
S_ps_c  = eval_power_series(t_check, s_float)
I_ps_c  = eval_power_series(t_check, i_float)
R_ps_c  = 1 - S_ps_c - I_ps_c

print("\n" + "="*95)
print(f"{'t':>6} | {'S_RK45':>14} | {'S_PS9':>14} | {'|err_S|':>14} "
      f"| {'I_RK45':>14} | {'I_PS9':>14} | {'|err_I|':>14} "
      f"| {'R_RK45':>14} | {'R_PS9':>14} | {'|err_R|':>14} ")
print("-"*95)
for k in range(len(t_check)):
    print(f"{t_check[k]:>6.1f} | {S_num_c[k]:>14.8f} | {S_ps_c[k]:>14.8f} | "
          f"{abs(S_ps_c[k]-S_num_c[k]):>14.8e} | "
          f"{I_num_c[k]:>14.8f} | {I_ps_c[k]:>14.8f} | "
          f"{abs(I_ps_c[k]-I_num_c[k]):>14.8e} | "
          f"{R_num_c[k]:>14.8f} | {R_ps_c[k]:>14.8f} | "
          f"{abs(R_ps_c[k]-R_num_c[k]):>14.8e}")
print("="*95)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette ──────────────────────────────────────────────────────────
CLR_RK  = "#1a6fb5"     # ODE45  – steel blue
CLR_PS  = "#e05c2a"     # PS9    – burnt orange
CLR_ES  = "#2a9e5c"     # error S – emerald
CLR_EI  = "#9b2ac9"     # error I – violet
BG      = "#f7f8fc"
GRID    = "#dde0e9"

plt.rcParams.update({
    "font.family"     : "DejaVu Serif",
    "axes.facecolor"  : BG,
    "figure.facecolor": "white",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.color"      : GRID,
    "grid.linewidth"  : 0.7,
})

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                        left=0.07, right=0.96, top=0.91, bottom=0.07)

# ── Panel A : S(t) ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_eval, S_num, color=CLR_RK,  lw=2.2, label="RK45 reference")
ax1.plot(t_eval, S_ps,  color=CLR_PS,  lw=1.8, ls="--", label="PS9 solution")
ax1.set_xlabel("$t$", fontsize=12)
ax1.set_ylabel("$S(t)$", fontsize=12)
ax1.set_title("Fraction for $S(t)$", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.margins(x=0)
ax1.set_xlim(0, tend)

# ── Panel B : I(t) ──────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_eval, I_num, color=CLR_RK,  lw=2.2, label="RK45 reference")
ax2.plot(t_eval, I_ps,  color=CLR_PS,  lw=1.8, ls="--", label="PS9 solution")
ax2.set_xlabel("$t$", fontsize=12)
ax2.set_ylabel("$I(t)$", fontsize=12)
ax2.set_title("Fraction for $I(t)$", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.margins(x=0)
ax2.set_xlim(0, tend)

# ── Panel D : R(t) ───────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t_eval, 1-S_num-I_num, color=CLR_RK,  lw=2.2, label="RK45 reference")
ax3.plot(t_eval, 1-S_ps-I_ps,  color=CLR_PS,  lw=1.8, ls="--", label="PS9 solution")
ax3.set_xlabel("$t$", fontsize=12)
ax3.set_ylabel("$R(t)$", fontsize=12)
ax3.set_title("Fraction for $R(t)$", fontsize=13, fontweight="bold")
ax3.legend(fontsize=10)
ax3.margins(x=0)
ax3.set_xlim(0, tend)

# ── Panel C : absolute errors (log scale) ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.semilogy(t_eval, err_S + 1e-18, color=CLR_ES, lw=1.8, 
             label=r"$|S_{\rm PS9} - S_{\rm RK45}|$")
ax4.semilogy(t_eval, err_I + 1e-18, color=CLR_EI, lw=1.8, ls="--",
             label=r"$|I_{\rm PS9} - I_{\rm RK45}|$")
ax4.semilogy(t_eval, err_R + 1e-18, color="black", lw=1.8, ls=":",
             label=r"$|R_{\rm PS9} - R_{\rm RK45}|$")
ax4.set_xlabel("$t$", fontsize=12)
ax4.set_ylabel("Absolute difference", fontsize=12)
ax4.set_title("Absolute difference", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10)
ax4.margins(x=0)
ax4.set_xlim(0, tend)
ax4.set_ylim(1e-18, 100)

plt.savefig("figure2.png", dpi=150, bbox_inches="tight")
plt.savefig("figure2.pdf", bbox_inches="tight")
plt.show()
print("\nFigures saved.")