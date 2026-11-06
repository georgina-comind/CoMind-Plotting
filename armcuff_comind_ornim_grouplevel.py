"""
Arm-cuff study: GROUP plots of relative blood flow for up to 10 participants.

Fill in the SUBJECTS list below (one block per participant), then run:
    python armcuff_group_plots.py

Plots saved to the GROUP_OUTPUT_DIR folder:
    group_comind_and_ornim.png   two panels: CoMind (EFS) on top, ORNIM below.
                                 Each participant is a thin line (% of their own
                                 baseline); the median over participants is a thick line.
    group_median_comparison.png  CoMind and ORNIM medians on one axis, with the
                                 spread between participants (25th-75th percentile) shaded.
Data saved alongside:
    group_timeseries.csv         % of baseline for every participant on a common time grid
    group_summary.csv            per participant: occlusion level and post-release peak

Time zero = start of the 3 min baseline:
    ORNIM : the first marker in that participant's Data-MRK.csv
    CoMind: "comind_start" seconds into the recording (default 0)
"""

import csv
import os
import sys
from datetime import datetime

import h5py
import matplotlib

matplotlib.use("Agg")  # save to files; no screen needed over SSH
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# PARTICIPANTS - edit this list (up to 10).
# For each participant give:
#   name        : label used in the plots
#   comind      : full path to the CoMind fits .hdf5 file
#   ornim_cfi   : full path to the ORNIM Data-CFI.csv file
#   ornim_mrk   : full path to the ORNIM Data-MRK.csv file
#   comind_start: (optional) seconds into the CoMind recording when the
#                 baseline started - leave out if the recording began at baseline
# Leave "ornim_cfi"/"ornim_mrk" as "" if a participant has no ORNIM data
# (or "comind" as "" if there is no CoMind data).
# =====================================================================
STUDY_DIR = "/knox2/TOOE/ORNIM/2026_11_03_ornimefs_armcuff"

SUBJECTS = [
    {
        "name": "ARM-020-002",
        "comind": f"{STUDY_DIR}/ARM-020-002/fits/fits_001.hdf5",
        "ornim_cfi": f"{STUDY_DIR}/ARM-020-002/3261152/Data-CFI.csv",
        "ornim_mrk": f"{STUDY_DIR}/ARM-020-002/3261152/Data-MRK.csv",
    },

    {
        "name": "ARM-020-003",
        "comind": f"{STUDY_DIR}/ARM-020-003/fits/fits_002.hdf5",
        "ornim_cfi": f"{STUDY_DIR}/ARM-020-003/3261154/Data-CFI.csv",
        "ornim_mrk": f"{STUDY_DIR}/ARM-020-003/3261154/Data-MRK.csv",
    },

    {
        "name": "ARM-020-004",
        "comind": f"{STUDY_DIR}/ARM-020-004/fits/fits_001.hdf5",
        "ornim_cfi": f"{STUDY_DIR}/ARM-020-004/3261155/Data-CFI.csv",
        "ornim_mrk": f"{STUDY_DIR}/ARM-020-004/3261155/Data-MRK.csv",
    },
    # Copy the block above for each extra participant, e.g.:
    # {
    #     "name": "ARM-020-003",
    #     "comind": f"{STUDY_DIR}/ARM-020-003/fits/fits_001.hdf5",
    #     "ornim_cfi": f"{STUDY_DIR}/ARM-020-003/<ornim folder>/Data-CFI.csv",
    #     "ornim_mrk": f"{STUDY_DIR}/ARM-020-003/<ornim folder>/Data-MRK.csv",
    #     "comind_start": 0,
    # },
]

MAX_SUBJECTS = 10

# Where the plots go. None = a "group_plots" folder next to this script.
GROUP_OUTPUT_DIR = None

# =====================================================================
# ANALYSIS SETTINGS
# =====================================================================
COMIND_TOF_NS = 1     # CoMind time-of-flight gate used for every participant (ns) - previously 0.48
ORNIM_CHANNEL = None     # ORNIM channel (1 or 2); None = whichever has the most valid data

BASELINE_END = 3 * 60    # protocol timings, seconds from baseline start
OCCLUSION_END = 6 * 60
RECOVERY_END = 15 * 60
BASELINE_REF = (0, 2 * 60)  # window averaged as "baseline" (avoids cuff-inflation artefact)

GRID_STEP = 3.0          # common time grid, seconds (ORNIM updates every ~3 s)
SHOW_IQR = True          # shade 25th-75th percentile around the median
Y_MAX_PERCENT = None     # cap on the y-axis in %, e.g. 600; None = automatic

COMIND_LABEL = "CoMind (EFS)"
ORNIM_LABEL = "ORNIM"
COMIND_MEDIAN_COLOR = "tab:blue"
ORNIM_MEDIAN_COLOR = "tab:red"


# =====================================================================
# HELPERS
# =====================================================================
GRID = np.arange(0, RECOVERY_END + GRID_STEP, GRID_STEP)  # common time axis (s)


def to_grid(t, y):
    """Average y into GRID bins (bin centred on each grid time). NaN where no data."""
    edges = np.concatenate([GRID - GRID_STEP / 2, [GRID[-1] + GRID_STEP / 2]])
    which = np.digitize(t, edges) - 1
    ok = (which >= 0) & (which < len(GRID)) & np.isfinite(y)
    sums = np.bincount(which[ok], weights=y[ok], minlength=len(GRID))
    counts = np.bincount(which[ok], minlength=len(GRID))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def to_percent_of_baseline(y_grid):
    base = (GRID >= BASELINE_REF[0]) & (GRID < BASELINE_REF[1])
    ref = np.nanmean(y_grid[base]) if np.any(np.isfinite(y_grid[base])) else np.nan
    if not np.isfinite(ref) or ref == 0:
        raise ValueError("no usable data in the baseline window")
    return 100 * y_grid / ref


def shade_phases(ax, legend=True):
    kw = lambda name: {"label": name} if legend else {}
    ax.axvspan(0, BASELINE_END / 60, color="tab:green", alpha=0.07, **kw("Baseline"))
    ax.axvspan(BASELINE_END / 60, OCCLUSION_END / 60, color="tab:red", alpha=0.09, **kw("Occlusion"))
    ax.axvspan(OCCLUSION_END / 60, RECOVERY_END / 60, color="tab:blue", alpha=0.05, **kw("Recovery"))
    for x in (BASELINE_END, OCCLUSION_END):
        ax.axvline(x / 60, color="k", lw=0.8, ls="--")
    ax.axhline(100, color="grey", lw=0.8)


# ---------------------------------------------------------------- CoMind
def load_comind(path, baseline_start):
    """Return CoMind xi as % of baseline on GRID, at the COMIND_TOF_NS gate."""
    with h5py.File(path, "r") as f:
        tofs = f["tofs"][:]
        i = int(np.argmin(np.abs(tofs * 1e9 - COMIND_TOF_NS)))
        t = f["times"][:] - baseline_start
        xi = f["xi"][:, i, 0].astype(float)
    upper = 5e5  # fit upper limit seen in the fits files
    bad = (xi <= 0) | (xi >= upper) | ~np.isfinite(xi)
    if bad.mean() > 0.1:
        print(f"    WARNING: {bad.mean() * 100:.0f}% of fits at ToF {tofs[i] * 1e9:.2f} ns are unusable")
    xi[bad] = np.nan
    return to_percent_of_baseline(to_grid(t, xi)), tofs[i] * 1e9


# ---------------------------------------------------------------- ORNIM
def parse_time(text):
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised time format: {text!r}")


def read_ornim_table(path):
    """ORNIM CSV: 2 lines of file info, a blank line, then the data table."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))
    blank = next(i for i, ln in enumerate(lines) if i > 0 and not any(x.strip() for x in ln))
    header = [h.strip() for h in lines[blank + 1]]
    return [{h: x.strip() for h, x in zip(header, ln) if h}
            for ln in lines[blank + 2:] if any(x.strip() for x in ln)]


def load_ornim(cfi_path, mrk_path):
    """Return ORNIM CFI as % of baseline on GRID, and the channel used."""
    t0 = min(parse_time(m["MarkerTimeDisplayed"]) for m in read_ornim_table(mrk_path))
    data = {}
    for r in read_ornim_table(cfi_path):
        if r.get("CRX_ValidityBits", "0") != "0":  # flagged invalid by ORNIM
            continue
        ch = int(r["CRX_Channel"])
        data.setdefault(ch, ([], []))
        data[ch][0].append((parse_time(r["CRX_Timestamp"]) - t0).total_seconds())
        data[ch][1].append(float(r["CRX_Value_CFI"]))
    if not data:
        raise ValueError("no valid CFI readings")
    ch = ORNIM_CHANNEL if ORNIM_CHANNEL is not None else max(data, key=lambda k: len(data[k][0]))
    if ch not in data:
        raise ValueError(f"channel {ch} has no valid readings (channels with data: {sorted(data)})")
    t, v = np.array(data[ch][0]), np.array(data[ch][1])
    order = np.argsort(t)
    t, v = t[order], v[order]
    # ORNIM updates every ~3-4 s, so interpolate onto the grid (no gaps), but
    # leave NaN outside the recording or where readings are > 15 s apart
    y = np.interp(GRID, t, v, left=np.nan, right=np.nan)
    nearest = np.min(np.abs(GRID[:, None] - t[None, :]), axis=1)
    y[nearest > 15] = np.nan
    return to_percent_of_baseline(y), ch


# ---------------------------------------------------------------- summary numbers
def summarise(y):
    """Mean % in the last minute of occlusion, and peak % in the first 3 min of recovery."""
    occ = (GRID >= OCCLUSION_END - 60) & (GRID < OCCLUSION_END)
    rec = (GRID >= OCCLUSION_END) & (GRID < OCCLUSION_END + 180)
    occ_mean = np.nanmean(y[occ]) if np.any(np.isfinite(y[occ])) else np.nan
    if np.any(np.isfinite(y[rec])):
        k = np.nanargmax(np.where(rec, y, np.nan))
        return occ_mean, y[k], (GRID[k] - OCCLUSION_END)
    return occ_mean, np.nan, np.nan


# =====================================================================
# PLOTTING
# =====================================================================
def plot_panel(ax, curves, title, median_color):
    shade_phases(ax)
    colors = plt.cm.tab10(np.arange(10))
    for k, (name, y) in enumerate(curves.items()):
        ax.plot(GRID / 60, y, color=colors[k % 10], lw=0.9, alpha=0.6, label=name)
    if curves:
        stack = np.vstack(list(curves.values()))
        with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
            q1, q3 = np.nanpercentile(stack, [25, 75], axis=0)
        if SHOW_IQR and len(curves) > 2:
            ax.fill_between(GRID / 60, q1, q3, color=median_color, alpha=0.15, label="25th-75th percentile")
        ax.plot(GRID / 60, med, color="black", lw=3.2)                       # outline
        ax.plot(GRID / 60, med, color=median_color, lw=2.2, label=f"Median (n={len(curves)})")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_ylabel("% of baseline")
    ax.set_ylim(0, Y_MAX_PERCENT)
    ax.legend(loc="upper left", fontsize=7, ncol=3)


def group_median(curves):
    stack = np.vstack(list(curves.values()))
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmedian(stack, axis=0), *np.nanpercentile(stack, [25, 75], axis=0)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# =====================================================================
# MAIN
# =====================================================================
def main():
    if not SUBJECTS:
        sys.exit("The SUBJECTS list is empty - add your participants at the top of the script.")
    if len(SUBJECTS) > MAX_SUBJECTS:
        sys.exit(f"{len(SUBJECTS)} participants listed - the maximum is {MAX_SUBJECTS}.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.abspath(GROUP_OUTPUT_DIR or os.path.join(script_dir, "group_plots"))
    os.makedirs(out, exist_ok=True)
    print(f"Saving group plots to: {out}\n")

    comind, ornim, summary = {}, {}, []
    tof_used = None

    for s in SUBJECTS:
        name = s["name"]
        print(f"--- {name}")
        row = {"name": name}

        if s.get("comind"):
            try:
                if not os.path.isfile(s["comind"]):
                    raise FileNotFoundError(s["comind"])
                comind[name], tof_used = load_comind(s["comind"], s.get("comind_start", 0.0))
                row["comind_occlusion_%"], row["comind_peak_%"], row["comind_peak_time_s"] = summarise(comind[name])
                print(f"    CoMind loaded (ToF {tof_used:.2f} ns)")
            except Exception as e:
                print(f"    !!! CoMind skipped: {type(e).__name__}: {e}")

        if s.get("ornim_cfi") and s.get("ornim_mrk"):
            try:
                for p in (s["ornim_cfi"], s["ornim_mrk"]):
                    if not os.path.isfile(p):
                        raise FileNotFoundError(p)
                ornim[name], ch = load_ornim(s["ornim_cfi"], s["ornim_mrk"])
                row["ornim_occlusion_%"], row["ornim_peak_%"], row["ornim_peak_time_s"] = summarise(ornim[name])
                print(f"    ORNIM loaded (channel {ch})")
            except Exception as e:
                print(f"    !!! ORNIM skipped: {type(e).__name__}: {e}")
        summary.append(row)

    print(f"\nLoaded CoMind for {len(comind)} and ORNIM for {len(ornim)} of {len(SUBJECTS)} participants\n")
    if not comind and not ornim:
        sys.exit("Nothing loaded - check the file paths in SUBJECTS.")

    comind_title = f"{COMIND_LABEL}: xi at ToF {tof_used:.2f} ns" if tof_used else COMIND_LABEL

    # ---- figure 1: two panels, all participants + median
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plot_panel(ax1, comind, comind_title, COMIND_MEDIAN_COLOR)
    plot_panel(ax2, ornim, f"{ORNIM_LABEL}: CFI", ORNIM_MEDIAN_COLOR)
    ax2.set_xlim(0, RECOVERY_END / 60)
    ax2.set_xlabel("Time since start of baseline (min)")
    fig.suptitle("Group response to arm-cuff occlusion (relative to baseline)", fontsize=13)
    save(fig, os.path.join(out, "group_comind_and_ornim.png"))

    # ---- figure 2: the two medians on one axis
    fig, ax = plt.subplots(figsize=(12, 5.5))
    shade_phases(ax)
    for curves, label, color in ((comind, COMIND_LABEL, COMIND_MEDIAN_COLOR),
                                 (ornim, ORNIM_LABEL, ORNIM_MEDIAN_COLOR)):
        if not curves:
            continue
        med, q1, q3 = group_median(curves)
        if SHOW_IQR and len(curves) > 2:
            ax.fill_between(GRID / 60, q1, q3, color=color, alpha=0.15)
        ax.plot(GRID / 60, med, color=color, lw=2.2, label=f"{label} median (n={len(curves)})")
    ax.set_xlim(0, RECOVERY_END / 60)
    ax.set_ylim(0, Y_MAX_PERCENT)
    ax.set_xlabel("Time since start of baseline (min)")
    ax.set_ylabel("% of baseline")
    ax.set_title("Median response: CoMind (EFS) vs ORNIM"
                 + (" (shaded = 25th-75th percentile)" if SHOW_IQR else ""))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save(fig, os.path.join(out, "group_median_comparison.png"))

    # ---- data files
    ts_path = os.path.join(out, "group_timeseries.csv")
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = [(f"comind_{n}", y) for n, y in comind.items()] + [(f"ornim_{n}", y) for n, y in ornim.items()]
        w.writerow(["time_s"] + [c for c, _ in cols])
        for k, t in enumerate(GRID):
            w.writerow([f"{t:.1f}"] + [("" if np.isnan(y[k]) else f"{y[k]:.2f}") for _, y in cols])
    print("Saved", ts_path)

    sum_path = os.path.join(out, "group_summary.csv")
    fields = ["name", "comind_occlusion_%", "comind_peak_%", "comind_peak_time_s",
              "ornim_occlusion_%", "ornim_peak_%", "ornim_peak_time_s"]
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary:
            w.writerow({k: (f"{v:.1f}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    print("Saved", sum_path)

    print("\nSummary (% of baseline):")
    print(f"  {'participant':<14}{'CoMind occl.':>14}{'CoMind peak':>13}{'ORNIM occl.':>13}{'ORNIM peak':>12}")
    for row in summary:
        g = lambda k: f"{row[k]:.0f}" if k in row and np.isfinite(row[k]) else "-"
        print(f"  {row['name']:<14}{g('comind_occlusion_%'):>14}{g('comind_peak_%'):>13}"
              f"{g('ornim_occlusion_%'):>13}{g('ornim_peak_%'):>12}")
    print("  (occl. = mean over the last minute of occlusion; peak = max in first 3 min of recovery)")


if __name__ == "__main__":
    main()