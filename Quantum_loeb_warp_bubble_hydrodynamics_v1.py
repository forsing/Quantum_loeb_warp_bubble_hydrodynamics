# Quantum_loeb_warp_bubble_hydrodynamics 



import random
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
FLOW_WINDOW = 5
WARP_BETA = 0.75
WALL_THICKNESS_GRID = (1.8, 3.2, 5.0)
STAGNATION_GAIN_GRID = (0.45, 0.90)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < FLOW_WINDOW + 35:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_qft_operators() -> tuple[np.ndarray, np.ndarray]:
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))
    u_qft = Operator(qc).data
    u_iqft = np.conj(u_qft.T)
    return u_qft, u_iqft


def fit_ambient_interstellar_medium(draws_train: np.ndarray) -> np.ndarray:
    n_tr = len(draws_train)
    occ = np.zeros((n_tr, MAX_NUM), dtype=np.float64)
    occ[np.arange(n_tr)[:, None], draws_train - 1] = 1.0
    ambient_rho = np.mean(occ, axis=0)
    return ambient_rho / max(float(np.sum(ambient_rho)), 1e-12)


def alcubierre_shape_function(r_dist: np.ndarray, radius_r: float, sigma_wall: float) -> np.ndarray:
    num = np.tanh(sigma_wall * (r_dist + radius_r)) - np.tanh(sigma_wall * (r_dist - radius_r))
    den = max(2.0 * float(np.tanh(sigma_wall * radius_r)), 1e-8)
    return np.clip(num / den, 0.0, 1.0)







def simulate_warp_bubble_hydrodynamics(
    draws: np.ndarray,
    ambient_rho: np.ndarray,
    u_qft: np.ndarray,
    u_iqft: np.ndarray,
    wall_sigma: float,
    stagnation_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    n_samples = n_total - FLOW_WINDOW
    n_all_win = n_samples + 1

    occ_all = np.zeros((n_total, MAX_NUM), dtype=np.float64)
    occ_all[np.arange(n_total)[:, None], draws - 1] = 1.0

    grid_x = np.arange(MAX_NUM, dtype=np.float64)
    lorentz_gamma = 1.0 / np.sqrt(1.0 - (WARP_BETA ** 2))

    flow_weights = np.exp(np.linspace(-1.3, 0.0, num=FLOW_WINDOW, dtype=np.float64))
    flow_weights = flow_weights / np.sum(flow_weights)

    q_init_batch = np.zeros((n_all_win, STATE_DIM), dtype=np.complex128)
    hydro_density_batch = np.empty((n_all_win, MAX_NUM), dtype=np.float64)
    flow_signs = np.empty(n_all_win, dtype=np.float64)

    for idx in range(n_all_win):
        win_draws = draws[idx : idx + FLOW_WINDOW].astype(np.float64) - 1.0
        win_occ = occ_all[idx : idx + FLOW_WINDOW]

        centers = np.mean(win_draws, axis=1)
        radii = np.maximum(np.std(win_draws, axis=1), 2.5)

        bubble_center = float(np.sum(centers * flow_weights))
        bubble_radius = float(np.sum(radii * flow_weights))
        drift_vel = float(centers.__getitem__(FLOW_WINDOW - 1) - centers.__getitem__(FLOW_WINDOW - 2))
        flow_sign = 1.0 if drift_vel >= 0.0 else -1.0
        flow_signs[idx] = flow_sign

        signed_offset = flow_sign * (grid_x - bubble_center)
        r_abs = np.abs(signed_offset)

        f_shape = alcubierre_shape_function(r_abs, radius_r=bubble_radius, sigma_wall=1.0 / wall_sigma)
        wall_grad = np.abs(np.gradient(f_shape))
        wall_grad = wall_grad / max(float(np.sum(wall_grad)), 1e-12)

        ahead_wall = np.maximum(signed_offset - bubble_radius, 0.0)
        stagnation_zone = np.exp(-0.5 * ((ahead_wall / max(0.45 * bubble_radius, 1.2)) ** 2)) * (signed_offset > 0.0)

        shock_nose_pos = bubble_radius + 0.55 * max(0.45 * bubble_radius, 1.2)
        bow_shock = np.exp(-0.5 * (((signed_offset - shock_nose_pos) / max(0.35 * wall_sigma, 0.8)) ** 2))

        local_inflow = np.sum(win_occ * flow_weights[:, None], axis=0)
        local_inflow = local_inflow / max(float(np.sum(local_inflow)), 1e-12)

        compressed_rho = (
            lorentz_gamma * ambient_rho * (1.0 + stagnation_gain * stagnation_zone + 0.85 * bow_shock)
            + 0.65 * local_inflow * (1.0 - 0.5 * f_shape)
            + 0.30 * wall_grad
        )
        compressed_rho = np.maximum(compressed_rho, 1e-12)
        hydro_density_batch[idx] = compressed_rho

        vel_field = WARP_BETA * (1.0 - f_shape) * (1.0 - 0.7 * stagnation_zone)
        phase_flow = np.exp(1j * np.pi * flow_sign * vel_field * lorentz_gamma)
        q_init_batch[idx, :MAX_NUM] = np.sqrt(compressed_rho) * phase_flow

    q_init_batch = q_init_batch / np.maximum(np.linalg.norm(q_init_batch, axis=1, keepdims=True), 1e-12)

    freq_idx = np.arange(STATE_DIM, dtype=np.float64)
    signed_k = np.where(freq_idx < float(STATE_DIM // 2), freq_idx, freq_idx - float(STATE_DIM))
    k_norm = signed_k / float(STATE_DIM // 2)
    advection_phase = np.exp(1j * WARP_BETA * np.pi * flow_signs[:, None] * k_norm[None, :])

    q_freq = np.einsum("ij,nj->ni", u_qft, q_init_batch) * advection_phase
    q_advected = np.einsum("ij,nj->ni", u_iqft, q_freq)
    q_dens = np.abs(q_advected[:, :MAX_NUM]) ** 2
    q_dens = q_dens / np.maximum(np.sum(q_dens, axis=1, keepdims=True), 1e-12)

    hydro_norm = hydro_density_batch / np.maximum(np.sum(hydro_density_batch, axis=1, keepdims=True), 1e-12)
    projected_rho = 0.60 * hydro_norm + 0.40 * q_dens
    projected_rho = projected_rho / np.maximum(np.sum(projected_rho, axis=1, keepdims=True), 1e-12)

    qc_verify = QuantumCircuit(NUM_QUBITS)
    qc_verify.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))
    qc_verify.append(QFTGate(NUM_QUBITS).inverse(), list(range(NUM_QUBITS)))
    _ = Statevector(q_init_batch.__getitem__(n_samples)).evolve(qc_verify)

    y_targets = draws[FLOW_WINDOW:].copy()
    return projected_rho[:n_samples], y_targets, projected_rho.__getitem__(n_samples)









def evaluate_warp_hydro_nll(
    rho_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    prob_mat = 0.87 * rho_eval + 0.13 * (1.0 / float(MAX_NUM))
    prob_mat = prob_mat / np.sum(prob_mat, axis=1, keepdims=True)

    n_eval = len(y_eval)
    target_mask = np.zeros((n_eval, MAX_NUM), dtype=bool)
    row_idx = np.arange(n_eval)[:, None]
    target_mask[row_idx, y_eval - 1] = True

    log_p = np.log(np.maximum(prob_mat, 1e-12))
    nll_per_draw = -np.sum(np.where(target_mask, log_p, 0.0), axis=1)
    return float(np.mean(nll_per_draw))


def decode_stagnation_shock_peaks(rho_vec: np.ndarray) -> list[int]:
    top_indices = np.argsort(rho_vec)[::-1][:NUM_POSITIONS]
    combo = sorted([int(idx) + 1 for idx in top_indices])
    if len(set(combo)) != NUM_POSITIONS:
        raise RuntimeError("Neuspesno dekodiranje 7 pikova gustine iz zone stagnacije warp mehura.")
    return combo


def main() -> None:
    print("Metoda: Avi Loeb Relativistic Warp Bubble Hydrodynamics Regressor (beta=0.75c, 1x6 qubita)")
    print("Posebnost: Alcubierre zid mehura + zona stagnacije + lucni udarni talas na Qiskit simulatoru")

    u_qft, u_iqft = build_qft_operators()

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        n_total = len(draws)
        n_samples = n_total - FLOW_WINDOW

        train_end = FLOW_WINDOW + int(0.75 * n_samples)
        val_end = FLOW_WINDOW + int(0.88 * n_samples)
        val_slice_start = train_end - FLOW_WINDOW
        val_slice_end = val_end - FLOW_WINDOW

        ambient_tr = fit_ambient_interstellar_medium(draws[:train_end])

        best_sigma = float(WALL_THICKNESS_GRID.__getitem__(0))
        best_stag = float(STAGNATION_GAIN_GRID.__getitem__(0))
        best_val_nll = float("inf")

        for w_sig in WALL_THICKNESS_GRID:
            for stag_g in STAGNATION_GAIN_GRID:
                rho_all, y_all, _ = simulate_warp_bubble_hydrodynamics(
                    draws,
                    ambient_rho=ambient_tr,
                    u_qft=u_qft,
                    u_iqft=u_iqft,
                    wall_sigma=w_sig,
                    stagnation_gain=stag_g,
                )
                val_nll = evaluate_warp_hydro_nll(
                    rho_all[val_slice_start:val_slice_end],
                    y_all[val_slice_start:val_slice_end],
                )
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_sigma = w_sig
                    best_stag = stag_g

        rho_eval_all, y_eval_all, _ = simulate_warp_bubble_hydrodynamics(
            draws,
            ambient_rho=ambient_tr,
            u_qft=u_qft,
            u_iqft=u_iqft,
            wall_sigma=best_sigma,
            stagnation_gain=best_stag,
        )
        test_nll = evaluate_warp_hydro_nll(
            rho_eval_all[val_slice_end:],
            y_eval_all[val_slice_end:],
        )

        ambient_full = fit_ambient_interstellar_medium(draws)
        _, _, next_rho = simulate_warp_bubble_hydrodynamics(
            draws,
            ambient_rho=ambient_full,
            u_qft=u_qft,
            u_iqft=u_iqft,
            wall_sigma=best_sigma,
            stagnation_gain=best_stag,
        )
        next_combo = decode_stagnation_shock_peaks(next_rho)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (wall_sigma={best_sigma}, stag_gain={best_stag}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Avi Loeb Relativistic Warp Bubble Hydrodynamics Regressor (beta=0.75c, 1x6 qubita)
Posebnost: Alcubierre zid mehura + zona stagnacije + lucni udarni talas na Qiskit simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (wall_sigma=5.0, stag_gain=0.45): 25.8488
Test NLL: 25.8616
NEXT: [3, x, 5, y, 7, z, 38]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (wall_sigma=5.0, stag_gain=0.45): 25.8374
Test NLL: 25.8714
NEXT: [6, x, 30, y, 33, z, 35]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (wall_sigma=5.0, stag_gain=0.45): 25.8813
Test NLL: 25.7937
NEXT: [1, x, 6, y, 8, z, 11]
"""



"""
A numerical hydrodynamic simulation of the air flow around a warp drive bubble, showing the projected density and velocity in the bubble's frame. 
For example, the warp bubble speed is 75% of the speed of light. 
The bubble wall is shown by the black circle near the shock nose. 
The stagnation zone is clearly displayed between the bubble wall and the tip of the shock zone, denoted by the blue zone in front of the black circle 
Avi Loeb 
warp drive 


Oblast za Avija Loeba i hidrodinamiku warp mehura (Alcubierre-Loeb Relativistic Hydrodynamic Flow Around a Warp Drive Bubble at \(\beta = v/c = 0.75\)) obuhvata relativističku fluidnu mehaniku oko zakrivljenog prostor-vremenskog zida mehura, gde se u referentnom sistemu mehura formiraju:
1. Zid mehura (Bubble Wall — ekspandirajuća/kontrahujuća metrika prostor-vremena).
2. Zona stagnacije (Stagnation Zone) neposredno ispred zida mehura gde se nadolazeći fluid usporava i sabija.
3. Lučni udarni talas (Bow Shock Nose) ispred zone stagnacije gde naglo skaču gustina i pritisak fluida.

Oblast relativističke hidrodinamike oko warp mehura (Alcubierreov profil zida mehura, zona stagnacije i lučni udarni talas pri \(\beta = 0.75\,c\)).

• Osnovna ideja (Avi Loeb — Relativistic Hydrodynamic Flow Around a Warp Bubble):
	• Kada se warp mehur kreće brzinom \(\beta = v/c = 0.75\) kroz medijum, u sistemu mehura javlja se Alcubierreov profil brzine pomeranja (shift vector) \(f(r_s)\) koji razdvaja unutrašnjost ravnog prostor-vremena od spoljašnjeg toka.
	• Između vrha udarnog talasa (bow shock) i zida mehura (bubble wall) nastaje zona stagnacije sa visokom lokalnom gustinom i niskom relativnom brzinom, dok iza zida mehura dolazi do razređivanja i ubrzavanja toka.

	1. 1D/ciklični relativistički hidrodinamički domen od 39 ćelija (na 1x6 qubita): Centar i radijus zida mehura za svako stanje određeni su iz težišta i rasipanja aktivnih brojeva u prozoru istorije (isključivo prošlost bez curenja budućnosti).
	2. Alcubierreov zid mehura + zona stagnacije + lučni udarni talas (QFTGate na simulatoru): Kvantno kolo na 6 qubita kombinuje gustinu stagnacije (sabijanje fluida ispred zida mehura), gradijent lučnog udarnog talasa (skok na čelu udara) i QFT advekciju toka u sistemu mehura pri \(\beta = 0.75\).
	3. Shock-Stagnation Peak Selection dekoder: NEXT kombinacija od 7 različitih brojeva dobija se direktnim izborom 7 ćelija sa najvećom relativističkom gustinom stagnacije i udarnog fronta (bez pos_bias matrica i bez DP lanca).

Modeluje dinamički pokretni zid mehura i zonu stagnacije ispred fronta umesto statičnih prelaza.
"""
