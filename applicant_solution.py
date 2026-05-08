import json
import gdown
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
if not Path(downloaded_file).exists() or Path(downloaded_file).stat().st_size == 0:
    gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n: np.ndarray, rx: np.ndarray) -> np.ndarray:
    ALPHA = 0.96
    BETA = 0.72
    RIDGE = 1e-6

    model_subset = slice(20_000, 220_000)
    lags: tuple[int, ...] = tuple(range(-5, 4))

    def band_matrix(x: np.ndarray):
        return np.column_stack([
            helpers['score_filter'](x[:, ch])
            for ch in range(x.shape[1])
        ])

    def shift_signal(x: np.ndarray, lag: int) -> np.ndarray:
        y = np.zeros_like(x)
        if lag >= 0:
            y[lag:] = x[:len(x) - lag]
        else:
            shift = -lag
            y[:len(x) - shift] = x[shift:]
        return y

    def shifted_window(x: np.ndarray, lag: int, start: int, stop: int) -> np.ndarray:
        return shift_signal(x, lag)[start:stop]

    def rank_component_from_vector(band: np.ndarray, vector: np.ndarray):
        shared = band @ vector
        denom = np.vdot(shared, shared) + 1e-20

        coeffs = np.array([
            np.vdot(shared, band[:, ch])/denom
            for ch in range(band.shape[1])
        ])

        return shared[:, None] * coeffs[None, :]

    def make_tx_terms() -> tuple[np.ndarray, ...]:
        score_filter = helpers['score_filter']

        return (
            score_filter(tx_n[:, 0] ** 2 * tx_n[:, 1].conj()),
            score_filter(tx_n[:, 1] ** 2 * tx_n[:, 0].conj()),
            score_filter(tx_n[:, 0] ** 2 * tx_n[:, 3].conj()),
            score_filter(tx_n[:, 3] ** 2 * tx_n[:, 0].conj()),
            score_filter(tx_n[:, 1] ** 2 * tx_n[:, 2].conj()),
            score_filter(tx_n[:, 2] ** 2 * tx_n[:, 1].conj()),
            score_filter(tx_n[:, 3] ** 2 * tx_n[:, 2].conj()),
            score_filter(tx_n[:, 2] ** 2 * tx_n[:, 3].conj()),
            score_filter(tx_n[:, 0] ** 2 * tx_n[:, 5].conj()),
            score_filter(tx_n[:, 5] ** 2 * tx_n[:, 0].conj()),
        )

    def fit_tx_prediction(
        inp: np.ndarray,
        tx_terms: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        start = model_subset.start
        stop = model_subset.stop

        model_x = np.column_stack([
            shifted_window(term, lag, start, stop)
            for term in tx_terms
            for lag in lags
        ])

        gram = model_x.conj().T @ model_x
        gram = gram + RIDGE*np.eye(gram.shape[0])

        pred = np.zeros_like(inp)

        for ch in range(inp.shape[1]):
            y = helpers['score_filter'](inp[:, ch])[model_subset]
            coef = np.linalg.solve(gram, model_x.conj().T @ y)
            coef = coef.reshape(len(tx_terms), len(lags))

            ch_pred = np.zeros(inp.shape[0], dtype=np.complex128)

            for term_idx, term in enumerate(tx_terms):
                for lag_idx, lag in enumerate(lags):
                    ch_pred += coef[term_idx, lag_idx]*shift_signal(term, lag)

            pred[:, ch] = ch_pred

        return pred

    raw_band = band_matrix(rx)

    cov = raw_band.conj().T @ raw_band / raw_band.shape[0]
    _, eigvecs = np.linalg.eigh(cov)

    rank1 = rank_component_from_vector(raw_band, eigvecs[:, -1])
    rank2 = rank_component_from_vector(raw_band, eigvecs[:, -2])

    rx_precleaned = rx - ALPHA*rank1 - BETA*rank2
    tx_terms = make_tx_terms()
    tx_prediction = fit_tx_prediction(rx_precleaned, tx_terms)
    rx_hat = rx_precleaned - tx_prediction

    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
