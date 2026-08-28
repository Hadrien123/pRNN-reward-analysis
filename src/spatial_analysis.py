import numpy as np
import pandas as pd
import torch
import pynapple as nap
from scipy.ndimage import gaussian_filter, maximum_filter, generate_binary_structure

from prnn.analysis.TuningCurveAnalysis import (
    RiaBWallGroups,
    calculateBorderScore,
    pf_autocorr,
    count_autocorr_peaks,
    calculate_field_size,
    calculate_field_asymmetry,
    TuningCurveAnalysis as TCA,
)

# ---------------------------------------------------------------------------
# Place-field computation
# ---------------------------------------------------------------------------


def calculateSpatialRepresentation(
    pRNN, env, obs, act, state, saveTrainingData=True, onsetTransient=20, activeTimeThreshold=200
):
    obs_pred, obs_next, h = pRNN.predict(obs, act)
    h = torch.mean(h, dim=0, keepdims=True)

    position = nap.TsdFrame(
        t=np.arange(onsetTransient, h.size(1)),
        d=state["agent_pos"][onsetTransient:-1, :],
        columns=("x", "y"),
        time_units="s",
    )
    rates = nap.TsdFrame(
        t=np.arange(onsetTransient, h.size(1)),
        d=h.squeeze().detach().numpy()[onsetTransient:, :],
        time_units="s",
    )

    nb_bins_x, nb_bins_y, minmax = env.get_map_bins()
    place_fields, xy = nap.compute_2d_tuning_curves_continuous(
        rates, position, ep=rates.time_support, nb_bins=(nb_bins_x, nb_bins_y), minmax=minmax
    )
    SI = nap.compute_2d_mutual_info(place_fields, position, position.time_support, bitssec=False)

    numactiveT = np.sum((h > 0).numpy(), axis=1)
    inactive_cells = numactiveT < activeTimeThreshold
    SI.iloc[inactive_cells.flatten()] = 0

    HD = nap.TsdFrame(
        t=np.arange(onsetTransient, h.size(1)),
        d=state["agent_dir"][onsetTransient:-1],
        columns=("HD",),
        time_units="s",
    )
    nb_bins, minmax = env.get_HD_bins()
    HD_tuningcurves = nap.compute_1d_tuning_curves_continuous(
        rates, HD, ep=rates.time_support, nb_bins=nb_bins, minmax=minmax
    )
    HD_info = nap.compute_1d_mutual_info(HD_tuningcurves, HD, HD.time_support, bitssec=False)
    SI["HDinfo"] = HD_info["SI"]

    if saveTrainingData:
        pRNN.addTrainingData("place_fields", place_fields)
        pRNN.addTrainingData("SI", SI["SI"])

    WAKEactivity = {
        "obs": obs,
        "act": act,
        "state": state,
        "obs_pred": obs_pred,
        "obs_next": obs_next,
        "h": np.squeeze(h.detach().numpy()),
    }
    return place_fields, SI, WAKEactivity


#modified from calculateSpatialRepresentation
def calculateInputSpatialRepresentation(
    pRNN, env, obs, act, state, saveTrainingData=True, onsetTransient=20, activeTimeThreshold=200
):
    #Input-channel (obs[2], the grid/place-cell input fed to the network) place fields.
    obs_used = obs[2]
    obs_pred, obs_next, h = pRNN.predict(obs, act)
    h = torch.mean(h, dim=0, keepdims=True)

    position = nap.TsdFrame(
        t=np.arange(onsetTransient, obs_used.size(1) - 1),
        d=state["agent_pos"][onsetTransient:-1, :],
        columns=("x", "y"),
        time_units="s",
    )
    rates = nap.TsdFrame(
        t=np.arange(onsetTransient, obs_used.size(1) - 1),
        d=obs_used.squeeze().detach().numpy()[onsetTransient:-1, :],
        time_units="s",
    )

    nb_bins_x, nb_bins_y, minmax = env.get_map_bins()
    place_fields, xy = nap.compute_2d_tuning_curves_continuous(
        rates, position, ep=rates.time_support, nb_bins=(nb_bins_x, nb_bins_y), minmax=minmax
    )
    SI = nap.compute_2d_mutual_info(place_fields, position, position.time_support, bitssec=False)

    numactiveT = np.sum((obs_used > 0).numpy(), axis=1)
    inactive_cells = numactiveT < activeTimeThreshold
    SI.iloc[inactive_cells.flatten()] = 0

    HD = nap.TsdFrame(
        t=np.arange(onsetTransient, obs_used.size(1) - 1),
        d=state["agent_dir"][onsetTransient:-1],
        columns=("HD",),
        time_units="s",
    )
    nb_bins, minmax = env.get_HD_bins()
    HD_tuningcurves = nap.compute_1d_tuning_curves_continuous(
        rates, HD, ep=rates.time_support, nb_bins=nb_bins, minmax=minmax
    )
    HD_info = nap.compute_1d_mutual_info(HD_tuningcurves, HD, HD.time_support, bitssec=False)
    SI["HDinfo"] = HD_info["SI"]

    if saveTrainingData:
        pRNN.addTrainingData("place_fields", place_fields)
        pRNN.addTrainingData("SI", SI["SI"])

    WAKEactivity = {
        "obs": obs,
        "act": act,
        "state": state,
        "obs_pred": obs_pred,
        "obs_next": obs_next,
        "h": np.squeeze(obs_used.detach().numpy()),
    }
    return place_fields, SI, WAKEactivity

#modified from calculateSpatialRepresentation
def calculateOutputSpatialRepresentation(
    pRNN, env, obs, act, state, saveTrainingData=True, onsetTransient=20, activeTimeThreshold=200
):
    #Predicted-output-channel (the network's own obs[2] prediction) place fields.
    obs_pred, obs_next, h = pRNN.predict(obs, act)
    obs_pred_used = obs_pred[1][0]
    h = torch.mean(h, dim=0, keepdims=True)

    position = nap.TsdFrame(
        t=np.arange(onsetTransient, obs_pred_used.size(1)),
        d=state["agent_pos"][onsetTransient:-1, :],
        columns=("x", "y"),
        time_units="s",
    )
    rates = nap.TsdFrame(
        t=np.arange(onsetTransient, obs_pred_used.size(1)),
        d=obs_pred_used.squeeze().detach().numpy()[onsetTransient:, :],
        time_units="s",
    )

    nb_bins_x, nb_bins_y, minmax = env.get_map_bins()
    place_fields, xy = nap.compute_2d_tuning_curves_continuous(
        rates, position, ep=rates.time_support, nb_bins=(nb_bins_x, nb_bins_y), minmax=minmax
    )
    SI = nap.compute_2d_mutual_info(place_fields, position, position.time_support, bitssec=False)

    numactiveT = np.sum((obs_pred_used > 0).numpy(), axis=1)
    inactive_cells = numactiveT < activeTimeThreshold
    SI.iloc[inactive_cells.flatten()] = 0

    HD = nap.TsdFrame(
        t=np.arange(onsetTransient, obs_pred_used.size(1)),
        d=state["agent_dir"][onsetTransient:-1],
        columns=("HD",),
        time_units="s",
    )
    nb_bins, minmax = env.get_HD_bins()
    HD_tuningcurves = nap.compute_1d_tuning_curves_continuous(
        rates, HD, ep=rates.time_support, nb_bins=nb_bins, minmax=minmax
    )
    HD_info = nap.compute_1d_mutual_info(HD_tuningcurves, HD, HD.time_support, bitssec=False)
    SI["HDinfo"] = HD_info["SI"]

    if saveTrainingData:
        pRNN.addTrainingData("place_fields", place_fields)
        pRNN.addTrainingData("SI", SI["SI"])

    WAKEactivity = {
        "obs": obs,
        "act": act,
        "state": state,
        "obs_pred": obs_pred,
        "obs_next": obs_next,
        "h": np.squeeze(obs_pred_used.detach().numpy()),
    }
    return place_fields, SI, WAKEactivity


# ---------------------------------------------------------------------------
# Cell-type classification
# ---------------------------------------------------------------------------


def calculateTuningCurveReliability(env, WAKEactivity, tuning_curves):
    FAKEactivity = TCA.makeFAKEdata(WAKEactivity, tuning_curves, start_pos=env.start_pos)
    TCreliability = FAKEactivity["TCcorr"]
    return FAKEactivity, TCreliability


def calculate_metrics(env, place_fields, SI, WAKEactivity):
    #Per-cell metrics used by groupCells. Cell order is fixed to sorted(place_fields.keys()).
    cell_ids = sorted(place_fields.keys())
    metrics = {
        "SI": SI.loc[cell_ids, "SI"].to_numpy(dtype=float),
        "HD_info": SI.loc[cell_ids, "HDinfo"].to_numpy(dtype=float),
    }
    FAKEactivity, metrics["EVs"] = calculateTuningCurveReliability(env, WAKEactivity, place_fields)
    wallgroups, not_near_walls = RiaBWallGroups(env)
    border_scores = [calculateBorderScore(place_fields[cid], wallgroups, not_near_walls) for cid in cell_ids]
    metrics["border_scores"] = np.array(border_scores)
    tc_autocorrs = pf_autocorr({cid: place_fields[cid] for cid in cell_ids}, peakNorm=True)
    autocorr_peaks = np.array([count_autocorr_peaks(ac) for ac in tc_autocorrs.values()])
    metrics["pf_peaks"] = (autocorr_peaks + 1) // 2
    metrics["fieldsize"] = np.array([calculate_field_size(ac) for ac in tc_autocorrs.values()])
    metrics["fieldasymmetry"] = np.array([calculate_field_asymmetry(ac) for ac in tc_autocorrs.values()])
    return metrics


def groupCells(
    metrics,
    SI_thresh=0.5,
    EV_unthresh=0.1,
    HD_thresh=0.35,
    border_thresh=0,
    EV_thresh=0.5,
    place_symmetrythresh=3,
    border_symmetrythresh=3,
):
    #Classify cells into untuned / HD / border / single-field / spatial+HD / complex.
    untuned = (metrics["EVs"] <= EV_unthresh) & (metrics["SI"] <= SI_thresh) & (metrics["HD_info"] <= HD_thresh)
    HD_cells = (metrics["EVs"] <= EV_unthresh) & (metrics["SI"] <= SI_thresh) & (metrics["HD_info"] > HD_thresh)

    border_cells = (
        ~untuned
        & ~HD_cells
        & (metrics["border_scores"] > border_thresh)
        & (metrics["fieldasymmetry"] > border_symmetrythresh)
    )

    single_field = (
        ~untuned
        & ~HD_cells
        & ~border_cells
        & (metrics["pf_peaks"] == 1)
        & (metrics["EVs"] > EV_thresh)
        & (metrics["fieldasymmetry"] < place_symmetrythresh)
    )

    spatial_HD = (
        ~untuned
        & ~HD_cells
        & ~border_cells
        & ~single_field
        & (metrics["SI"] > SI_thresh)
        & (metrics["HD_info"] > HD_thresh)
    )

    complex_cells = ~border_cells & ~single_field & ~untuned & ~HD_cells & ~spatial_HD

    groups = {
        "untuned": untuned,
        "HD_cells": HD_cells,
        "single_field": single_field,
        "border_cells": border_cells,
        "spatial_HD": spatial_HD,
        "complex_cells": complex_cells,
    }
    groupID = np.argmax(np.column_stack(list(groups.values())), axis=1)
    return groups, groupID


# ---------------------------------------------------------------------------
# Peak / circle / reward-proximity geometry
# ---------------------------------------------------------------------------


def peak_center(tc):
    #Return (row, col) of the highest-firing bin in a 2D tuning curve, or None if all-NaN.
    tc = np.asarray(tc, dtype=float)
    if tc.size == 0 or np.all(np.isnan(tc)):
        return None
    idx = np.nanargmax(tc)
    r, c = np.unravel_index(idx, tc.shape)
    return (int(r), int(c))


def centers_xy_from_rc(tuning_curves, idx, env, start_pos=0, use_start_offset=True):
    #Map cell indices -> (x, y) spatial coordinates of each cell's peak firing bin.
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=float)

    rc_list, keep_idx = [], []
    for i in idx:
        rc = peak_center(tuning_curves[int(i)])
        if rc is None:
            continue
        rc_list.append(rc)
        keep_idx.append(int(i))
    if not rc_list:
        return np.zeros((0, 2), dtype=float)

    rc = np.array(rc_list, dtype=int)
    coords = np.asarray(env.env.discrete_coords)
    H, W = coords.shape[:2]
    off = int(start_pos) if use_start_offset else 0
    rr = rc[:, 0] + off
    cc = rc[:, 1] + off
    mask = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return coords[rr[mask], cc[mask]].astype(float)


def _to_2d(a):
    #Return a 2-D float array from array-like or dict->{array}.
    if a is None or (isinstance(a, float) and np.isnan(a)):
        raise ValueError("missing")
    if isinstance(a, dict):
        a = a[0] if 0 in a else a[min(a.keys())]
    a = np.asarray(a, dtype=float).squeeze()
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[..., 0] if a.shape[-1] <= 4 else a[0, ...]
    raise ValueError(f"expected 2D array, got {a.shape}")


def _normalize(a):
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _peak_coords(a2d, threshold=0.3, neighborhood=2, min_distance=3, sigma=0.8):
    #Local-maxima peak detection combining raw and Gaussian-smoothed arrays.
    raw_arr = np.array(a2d)
    smooth_arr = gaussian_filter(a2d, sigma=sigma)
    fp = generate_binary_structure(raw_arr.ndim, neighborhood)

    raw_peaks = np.argwhere((raw_arr == maximum_filter(raw_arr, footprint=fp)) & (raw_arr >= threshold))
    smooth_peaks = np.argwhere((smooth_arr == maximum_filter(smooth_arr, footprint=fp)) & (raw_arr >= threshold))
    all_peaks = np.unique(np.vstack([raw_peaks, smooth_peaks]), axis=0)

    peak_values = raw_arr[all_peaks[:, 0], all_peaks[:, 1]]
    all_peaks = all_peaks[np.argsort(-peak_values)]

    filtered_peaks = []
    for peak in all_peaks:
        if not filtered_peaks:
            filtered_peaks.append(peak)
            continue
        distances = np.sqrt(np.sum((np.array(filtered_peaks) - peak) ** 2, axis=1))
        if np.all(distances > min_distance):
            filtered_peaks.append(peak)
    return [tuple(p) for p in filtered_peaks]


def identify_peaks(cells, *, threshold=0.6, neighborhood=1, normalize=True, cell_ids=None, sigma=0.8, min_distance=3):
    #Per-cell multi-peak detection over a list/array of 2D tuning curves. Thresholds tuned empirically.
    out = {}
    for i, cell in enumerate(cells):
        try:
            a = _to_2d(cell)
        except ValueError:
            continue
        if normalize:
            a = _normalize(a)
        key = cell_ids[i] if (cell_ids is not None and i < len(cell_ids)) else i
        out[key] = _peak_coords(a, threshold=threshold, neighborhood=neighborhood, min_distance=min_distance, sigma=sigma)
    return out


def _grid_xy(env, H, W):
    #Return (H,W,2) array of bin centers (x,y).
    e = getattr(env, "env", env)
    coords = getattr(e, "discrete_coords", None)
    if coords is not None and np.asarray(coords).shape[:2] == (H, W):
        return np.asarray(coords)
    xedges, yedges = getattr(e, "xedges", None), getattr(e, "yedges", None)
    if xedges is not None and yedges is not None:
        xc = (np.asarray(xedges)[:-1] + np.asarray(xedges)[1:]) / 2
        yc = (np.asarray(yedges)[:-1] + np.asarray(yedges)[1:]) / 2
        X, Y = np.meshgrid(xc, yc)
        return np.stack([X, Y], -1)
    X, Y = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    return np.stack([X, Y], -1)


def peaks_to_xy_centers(peaks_by_cell, pf_dict, env):
    #Convert per-cell (row, col) peak lists into (x, y) spatial coordinates.
    out = {}
    if not peaks_by_cell:
        return out
    first = next(pf_dict[k] for k in peaks_by_cell if k in pf_dict)
    H, W = _to_2d(first).shape
    grid = _grid_xy(env, H, W)

    for cid, rc_list in peaks_by_cell.items():
        if cid not in pf_dict or not rc_list:
            out[cid] = []
            continue
        xy = []
        for r, c in rc_list:
            x, y = grid[int(r), int(c)]
            xy.append((float(x), float(y)))
        out[cid] = xy
    return out


def counts_within_radius(points_xy, centers_xy, radius):
    points_xy = np.asarray(points_xy, dtype=float)
    centers_xy = np.asarray(centers_xy, dtype=float)
    if len(centers_xy) == 0:
        return np.array([], dtype=int)
    if len(points_xy) == 0:
        return np.zeros(len(centers_xy), dtype=int)
    diff = points_xy[None, :, :] - centers_xy[:, None, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.sum(d2 <= radius**2, axis=1).astype(int)


def build_circles(points_xy, reward_xy, *, radius=0.05):
    #Fraction of cells (in points_xy) landing within `radius` of each reward location.
    counts = counts_within_radius(points_xy, reward_xy, radius)
    total = max(len(points_xy), 1)
    return {
        "counts": [int(c) for c in counts],
        "fractions": [float(c) / float(total) for c in counts],
        "radius": float(radius),
    }


# ---------------------------------------------------------------------------
# Input-output place-field correlation
# ---------------------------------------------------------------------------


def take_correlation_2d(a, b, env_mask=None, window_size=5):
    #for 2D inputs, returns windowed local-correlation map
    #for 1D inputs, returns a single global Pearson correlation
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    if a.shape != b.shape:
        raise ValueError("Inputs must have the same shape.")

    if a.ndim == 1:
        x, y = a.ravel(), b.ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            return np.nan
        x, y = x[valid], y[valid]
        if np.std(x) == 0 or np.std(y) == 0:
            return 0  # constant signal -> 0 is more interpretable than NaN here
        return float(np.corrcoef(x, y)[0, 1])

    if env_mask is not None:
        a[~env_mask] = np.nan
        b[~env_mask] = np.nan

    corr_map = np.full(a.shape, np.nan)
    half_window = window_size // 2
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if env_mask is not None and not env_mask[i, j]:
                continue
            i0, i1 = max(0, i - half_window), min(a.shape[0], i + half_window + 1)
            j0, j1 = max(0, j - half_window), min(a.shape[1], j + half_window + 1)
            win_a, win_b = a[i0:i1, j0:j1].ravel(), b[i0:i1, j0:j1].ravel()
            valid = np.isfinite(win_a) & np.isfinite(win_b)
            if valid.sum() < 2:
                corr_map[i, j] = np.nan
            else:
                x, y = win_a[valid], win_b[valid]
                corr_map[i, j] = 0.0 if (np.std(x) == 0 or np.std(y) == 0) else np.corrcoef(x, y)[0, 1]
    return corr_map


# ---------------------------------------------------------------------------
# Reward-cell classification (Yaghoubi et al., 2026 shuffle test)
# ---------------------------------------------------------------------------


def classify_reward_cells(activity, positions, reward_positions, radius=0.05, n_shuffles=1000, percentile=99, rng=None):
    """
    Classify reward cells using a circular-shift shuffle control (Yaghoubi et al., 2026).

    For each cell, mean activity while the agent is within `radius` of any reward
    location is compared to the distribution of mean activities from `n_shuffles`
    circular shifts of the activity timeseries. A cell is a reward cell if its true
    mean exceeds the `percentile`-th percentile of the shuffle distribution.
    """

    activity = np.asarray(activity, dtype=float)
    positions = np.asarray(positions, dtype=float)
    reward_positions = np.atleast_2d(np.asarray(reward_positions, dtype=float))

    T, N = activity.shape
    T_use = min(T, positions.shape[0])
    activity = activity[:T_use]
    positions = positions[:T_use]

    dists = np.sqrt(((positions[:, None, :] - reward_positions[None, :, :]) ** 2).sum(axis=2))
    near_reward_mask = dists.min(axis=1) <= radius
    if near_reward_mask.sum() == 0:
        raise ValueError(
            f"No timesteps found within radius={radius} of reward positions. "
            "Check that positions and reward_positions share the same coordinate space."
        )

    true_mean = activity[near_reward_mask].mean(axis=0)

    if rng is None:
        rng = np.random.default_rng()
    shifts = rng.integers(1, T_use, size=n_shuffles)
    shuffle_means = np.empty((n_shuffles, N), dtype=float)
    for k, shift in enumerate(shifts):
        shuffled = np.roll(activity, int(shift), axis=0)
        shuffle_means[k] = shuffled[near_reward_mask].mean(axis=0)

    shuffle_threshold = np.percentile(shuffle_means, percentile, axis=0)
    is_reward_cell = true_mean > shuffle_threshold
    return is_reward_cell, true_mean, shuffle_threshold, near_reward_mask


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

import re


def parse_net_name(name):
    #Extract (n_repeats, recurrence) from a net filename like '..._100_0.05-...'.
    m = re.search(r"_(\d+)_(0?\.\d+|\d+(?:\.\d+)?)[-_]", name)
    return {
        "n_repeats": int(m.group(1)) if m else None,
        "recurrence": float(m.group(2)) if m else None,
    }
