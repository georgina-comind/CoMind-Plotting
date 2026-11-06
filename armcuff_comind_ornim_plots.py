"""
Arm-cuff study: CoMind BFI (xi) and ORNIM CFI, plotted separately and together.

Protocol: 3 min baseline, 3 min cuff occlusion, 9 min recovery.

Plots saved (all with baseline / occlusion / recovery shaded):
    1_comind_xi_vs_time.png     CoMind xi vs time at several ToFs
    1_comind_rBFI_vs_time.png   CoMind xi relative to baseline
    2_ornim_cfi_vs_time.png     ORNIM CFI vs time
    3_overlay_comind_ornim.png  CoMind and ORNIM together (relative + raw)
    extra_comind_heatmap.png    CoMind relative xi across all ToFs
    extra_comind_fit_quality.png

Time zero = start of the 3 min baseline:
    ORNIM : the first marker in Data-MRK.csv
    CoMind: COMIND_BASELINE_START seconds into the recording (default 0)

Run (with the .venv activated):
    python armcuff_comind_ornim_plots.py
Options:
    --subject ARM-020-003        another participant
    --tofs 0.3 0.6               CoMind ToFs to plot (ns)
    --overlay-tof 0.6            CoMind ToF used in the overlay (ns)
    --comind-start 12.5          baseline began 12.5 s into the CoMind recording
    --ornim-dir /path/to/folder  folder with Data-CFI.csv and Data-MRK.csv
    --out /path/to/folder        where to save the plots
"""

import argparse
import csv
import glob
import os
import sys
import traceback
from datetime import datetime

import h5py
import matplotlib

matplotlib.use("Agg")  # save to files; no screen needed over SSH
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_VERSION = "2.0 (CoMind + ORNIM)"

# =============================================================== SETTINGS
STUDY_DIR = "/knox2/TOOE/ORNIM/2026_11_03_ornimefs_armcuff"
DEFAULT_SUBJECT = "ARM-020-004"
COMIND_FILE = "fits/fits_001.hdf5"          # relative to the subject folder
ORNIM_SUBFOLDER = "3261155"                 # ORNIM folder inside the subject folder
ORNIM_CFI_FILE = "Data-CFI.csv"
ORNIM_MRK_FILE = "Data-MRK.csv"

COMIND_BASELINE_START = 0.0  # seconds into CoMind recording when baseline began

BASELINE_END = 3 * 60        # protocol timings, seconds from baseline start
OCCLUSION_END = 6 * 60
RECOVERY_END = 15 * 60
BASELINE_REF = (0, 2 * 60)   # window averaged as "baseline" (avoids ~2.4 min artefact)

BIN_SECONDS = 1.0            # CoMind smoothing for the CoMind plots
OVERLAY_BIN_SECONDS = 3.0    # CoMind smoothing for the overlay (ORNIM updates every ~3 s)
HEATMAP_BIN_SECONDS = 5.0
N_AUTO_TOFS = 6

COMIND_COLOR = "tab:blue"
ORNIM_COLORS = {1: "tab:orange", 2: "tab:red"}


# =============================================================== GENERAL HELPERS
def bin_average(t, y, width):
    """Average y in windows of `width` seconds, ignoring NaNs."""
    edges = np.arange(t[0], t[-1] + width, width)
    which = np.digitize(t, edges) - 1
    n = len(edges) - 1
    sums = np.bincount(which, weights=np.nan_to_num(y), minlength=n + 1)[:n]
    counts = np.bincount(which, weights=np.isfinite(y), minlength=n + 1)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        return edges[:-1] + width / 2, sums / counts


def baseline_mean(t, y):
    base = (t >= BASELINE_REF[0]) & (t < BASELINE_REF[1]) & np.isfinite(y)
    if not base.any():
        print("WARNING: no data in the baseline window; relative values not available")
        return np.nan
    return y[base].mean()


def shade_phases(ax, legend=True):
    kw = lambda name: {"label": name} if legend else {}
    ax.axvspan(0, BASELINE_END / 60, color="tab:green", alpha=0.08, **kw("Baseline"))
    ax.axvspan(BASELINE_END / 60, OCCLUSION_END / 60, color="tab:red", alpha=0.10, **kw("Occlusion"))
    ax.axvspan(OCCLUSION_END / 60, RECOVERY_END / 60, color="tab:blue", alpha=0.06, **kw("Recovery"))
    for x in (BASELINE_END, OCCLUSION_END):
        ax.axvline(x / 60, color="k", lw=0.8, ls="--")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# =============================================================== COMIND
def describe_comind(path):
    print("=" * 70)
    print("COMIND FILE:", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")
    print("=" * 70)
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            print(f"  attribute {k} = {v}")
        f.visititems(lambda name, obj: print(
            f"  {name:<8} shape={obj.shape} dtype={obj.dtype}" if isinstance(obj, h5py.Dataset)
            else f"  group {name}"))
        t, tofs, lags = f["times"][:], f["tofs"][:], f["lags"][:]
    dt = np.median(np.diff(t))
    print(f"  times: {t[0]:.2f}-{t[-1]:.2f} s ({t[-1] / 60:.1f} min), {1 / dt:.1f} Hz")
    print(f"  tofs : {len(tofs)} gates, {tofs[0] * 1e9:.2f}-{tofs[-1] * 1e9:.2f} ns")
    print(f"  lags : {len(lags)} lags, {lags[0] * 1e6:.1f}-{lags[-1] * 1e6:.1f} us")
    print("  xi[..., 0] = decay rate (1/s, BFI); xi[..., 1] = second fit parameter\n")


def load_comind(path, baseline_start):
    with h5py.File(path, "r") as f:
        times = f["times"][:] - baseline_start
        tofs = f["tofs"][:]
        xi = f["xi"][:, :, 0].copy()
        param2 = f["xi"][:, :, 1]

    # fits stuck at the limits (0 or the upper bound) are not real values
    xi[(xi <= 0) | (xi >= np.nanmax(xi))] = np.nan

    quality = np.mean(np.isfinite(xi), axis=0)
    signal = np.nanmedian(param2, axis=0)
    quality[signal < 0.01 * np.nanmax(signal)] = 0  # almost no light in this gate

    peak = int(np.nanargmax(signal))
    good = np.array([i for i in np.where(quality > 0.9)[0] if i >= peak])

    print("Usable fits per ToF:")
    for tof, q in zip(tofs, quality):
        print(f"  {tof * 1e9:6.2f} ns : {q * 100:5.1f}%" + ("" if q > 0.9 else "   (ignored)"))
    print(f"Usable ToFs for BFI: {tofs[good[0]] * 1e9:.2f} to {tofs[good[-1]] * 1e9:.2f} ns\n")
    return {"times": times, "tofs": tofs, "xi": xi, "param2": param2, "quality": quality, "good": good}


def tof_index(tofs, ns):
    return int(np.argmin(np.abs(tofs * 1e9 - ns)))


def plot_comind(c, idx, path, title, relative=False):
    t, tofs, xi = c["times"], c["tofs"], c["xi"]
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_phases(ax)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(idx)))
    lines = []
    for col, i in zip(colors, idx):
        y = xi[:, i]
        if relative:
            y = y / baseline_mean(t, y)
        else:
            ax.plot(t / 60, y, color=col, alpha=0.12, lw=0.5)
        tb, yb = bin_average(t, y, BIN_SECONDS)
        ax.plot(tb / 60, yb, color=col, lw=1.5, label=f"ToF {tofs[i] * 1e9:.2f} ns")
        lines.append(yb)
    lo, hi = np.nanpercentile(np.concatenate(lines), [0.5, 99.5])
    if relative:
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_ylim(0, hi * 1.1)
        ax.set_ylabel("rBFI (xi / baseline xi)")
    else:
        ax.set_yscale("log")
        ax.set_ylim(lo / 1.5, hi * 1.5)
        ax.set_ylabel("xi, decay rate (1/s) ~ BFI")
    ax.set_xlim(0, max(t[-1], RECOVERY_END) / 60)
    ax.set_xlabel("Time since start of baseline (min)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save(fig, path)


def plot_comind_heatmap(c, path, title):
    t, tofs, xi, good = c["times"], c["tofs"], c["xi"], c["good"]
    tb = bin_average(t, xi[:, good[0]], HEATMAP_BIN_SECONDS)[0]
    rows = []
    for i in good:
        yb = bin_average(t, xi[:, i], HEATMAP_BIN_SECONDS)[1]
        rows.append(yb / baseline_mean(tb, yb))
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(tb / 60, tofs[good] * 1e9, np.array(rows), shading="auto",
                       cmap="RdBu_r", vmin=0, vmax=2)
    for x in (BASELINE_END, OCCLUSION_END):
        ax.axvline(x / 60, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Time since start of baseline (min)")
    ax.set_ylabel("ToF (ns), later = deeper")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="xi / baseline")
    save(fig, path)


def plot_comind_quality(c, path, title):
    tofs = c["tofs"] * 1e9
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(tofs, c["quality"] * 100, width=0.05)
    axes[0].set_ylabel("Usable fits (%)")
    axes[1].semilogy(tofs, np.nanmedian(c["xi"], axis=0), "o-")
    axes[1].set_ylabel("Median xi (1/s)")
    axes[2].semilogy(tofs, np.nanmedian(c["param2"], axis=0), "o-")
    axes[2].set_ylabel("Median fit parameter 2")
    for ax in axes:
        ax.set_xlabel("ToF (ns)")
    fig.suptitle(title)
    save(fig, path)


# =============================================================== ORNIM
def find_ornim(subject, ornim_dir):
    """Return (CFI path, marker path)."""
    folder = ornim_dir or os.path.join(STUDY_DIR, subject, ORNIM_SUBFOLDER)
    cfi, mrk = os.path.join(folder, ORNIM_CFI_FILE), os.path.join(folder, ORNIM_MRK_FILE)
    if os.path.isfile(cfi) and os.path.isfile(mrk):
        return cfi, mrk

    root = os.path.join(STUDY_DIR, subject)
    print(f"ORNIM files not found in {folder}; searching {root} ...")
    paths = []
    for name in (ORNIM_CFI_FILE, ORNIM_MRK_FILE):
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        if not hits:
            raise FileNotFoundError(f"No {name} found under {root}. Use --ornim-dir to point to it.")
        paths.append(hits[0])
    return paths[0], paths[1]


def parse_time(text):
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised time format: {text!r}")


def read_ornim_table(path):
    """ORNIM CSV: 2 lines of file info, a blank line, then the data table."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))
    blank = next(i for i, ln in enumerate(lines) if i > 0 and not any(x.strip() for x in ln))
    header = [h.strip() for h in lines[blank + 1]]
    rows = []
    for ln in lines[blank + 2:]:
        if any(x.strip() for x in ln):
            rows.append({h: x.strip() for h, x in zip(header, ln) if h})
    return rows


def load_ornim(cfi_path, mrk_path):
    """Return {channel: (seconds since baseline start, CFI values)}."""
    markers = read_ornim_table(mrk_path)
    t0 = min(parse_time(m["MarkerTimeDisplayed"]) for m in markers)
    print("ORNIM markers:")
    for m in markers:
        print(f"   {m['MarkerTimeDisplayed']}  {m.get('MarkerLegend', '')}")
    print(f"   baseline start = {t0:%H:%M:%S} (first marker)")

    data, skipped = {}, 0
    for r in read_ornim_table(cfi_path):
        if r.get("CRX_ValidityBits", "0") != "0":  # non-zero = flagged invalid by ORNIM
            skipped += 1
            continue
        ch = int(r["CRX_Channel"])
        data.setdefault(ch, ([], []))
        data[ch][0].append((parse_time(r["CRX_Timestamp"]) - t0).total_seconds())
        data[ch][1].append(float(r["CRX_Value_CFI"]))

    if not data:
        raise ValueError(f"No valid CFI readings in {cfi_path}")
    channels = {ch: (np.array(t), np.array(v)) for ch, (t, v) in sorted(data.items())}
    print(f"   skipped {skipped} invalid readings")
    for ch, (t, v) in channels.items():
        print(f"   channel {ch}: {len(t)} readings, {t[0] / 60:.1f}-{t[-1] / 60:.1f} min, CFI {v.min():.0f}-{v.max():.0f}")
    print()
    return channels


def plot_ornim(channels, path, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_phases(ax)
    for ch, (t, v) in channels.items():
        ax.plot(t / 60, v, "o-", ms=2.5, lw=1.2, color=ORNIM_COLORS.get(ch), label=f"ORNIM channel {ch}")
    ax.set_xlim(0, max(max(t[-1] for t, _ in channels.values()), RECOVERY_END) / 60)
    ax.set_xlabel("Time since start of baseline (min)")
    ax.set_ylabel("CFI (ORNIM units)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save(fig, path)


# =============================================================== OVERLAY
def plot_overlay(c, tof_i, channels, path, title):
    tb, xb = bin_average(c["times"], c["xi"][:, tof_i], OVERLAY_BIN_SECONDS)
    comind_label = f"CoMind xi, ToF {c['tofs'][tof_i] * 1e9:.2f} ns"
    t_end = max([tb[-1], RECOVERY_END] + [t[-1] for t, _ in channels.values()])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # top: each system as % of its own baseline
    shade_phases(ax1)
    ax1.plot(tb / 60, 100 * xb / baseline_mean(tb, xb), color=COMIND_COLOR, lw=1.3, label=comind_label)
    for ch, (t, v) in channels.items():
        ax1.plot(t / 60, 100 * v / baseline_mean(t, v), "o-", ms=2.5, lw=1.3,
                 color=ORNIM_COLORS.get(ch), label=f"ORNIM CFI, channel {ch}")
    ax1.axhline(100, color="grey", lw=0.8)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("% of baseline")
    ax1.set_title(title + ": relative to baseline")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)

    # bottom: raw values on separate axes
    shade_phases(ax2, legend=False)
    ax2.plot(tb / 60, xb, color=COMIND_COLOR, lw=1.3)
    ax2.set_ylabel("CoMind xi (1/s)", color=COMIND_COLOR)
    ax2.tick_params(axis="y", colors=COMIND_COLOR)
    ax2b = ax2.twinx()
    for ch, (t, v) in channels.items():
        ax2b.plot(t / 60, v, "o-", ms=2.5, lw=1.3, color=ORNIM_COLORS.get(ch))
    ax2b.set_ylabel("ORNIM CFI", color="tab:red")
    ax2b.tick_params(axis="y", colors="tab:red")
    ax2.set_title("Raw values (separate axes)")
    ax2.set_xlim(0, t_end / 60)
    ax2.set_xlabel("Time since start of baseline (min)")
    save(fig, path)


# =============================================================== MAIN
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default=DEFAULT_SUBJECT)
    p.add_argument("--file", help="CoMind fits .hdf5 (default: <study>/<subject>/fits/fits_001.hdf5)")
    p.add_argument("--ornim-dir", help="folder with Data-CFI.csv and Data-MRK.csv")
    p.add_argument("--tofs", type=float, nargs="+", help="CoMind ToFs to plot, ns")
    p.add_argument("--overlay-tof", type=float, help="CoMind ToF for the overlay, ns")
    p.add_argument("--comind-start", type=float, default=COMIND_BASELINE_START,
                   help="seconds into CoMind recording when baseline began")
    p.add_argument("--out", help="output folder (default: plots_<subject> next to this script)")
    args = p.parse_args()

    print(f"armcuff_comind_ornim_plots.py  version {SCRIPT_VERSION}\n")

    out = os.path.abspath(args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   f"plots_{args.subject}"))
    os.makedirs(out, exist_ok=True)
    print(f"Saving all plots to: {out}\n")
    name = args.subject

    # ---- CoMind
    comind_path = args.file or os.path.join(STUDY_DIR, args.subject, COMIND_FILE)
    if not os.path.isfile(comind_path):
        sys.exit(f"CoMind file not found: {comind_path}  (use --file to point to it)")
    describe_comind(comind_path)
    c = load_comind(comind_path, args.comind_start)

    if args.tofs:
        idx = sorted({tof_index(c["tofs"], ns) for ns in args.tofs})
    else:
        g = c["good"]
        idx = sorted(set(g[np.linspace(0, len(g) - 1, N_AUTO_TOFS).round().astype(int)]))
    overlay_i = tof_index(c["tofs"], args.overlay_tof) if args.overlay_tof else int(c["good"][len(c["good"]) // 2])

    # ---- ORNIM (loaded before plotting so any problem shows up early)
    print("=" * 70)
    try:
        cfi_path, mrk_path = find_ornim(args.subject, args.ornim_dir)
        print("ORNIM FILES:\n  ", cfi_path, "\n  ", mrk_path)
        print("=" * 70)
        channels = load_ornim(cfi_path, mrk_path)
    except Exception:
        print("\n!!! ORNIM data could not be loaded, so plots 2 and 3 will be skipped:\n")
        traceback.print_exc()
        channels = None

    # ---- plots
    plot_comind(c, idx, os.path.join(out, "1_comind_xi_vs_time.png"), f"{name}: CoMind xi vs time")
    plot_comind(c, idx, os.path.join(out, "1_comind_rBFI_vs_time.png"), f"{name}: CoMind relative BFI",
                relative=True)
    if channels:
        plot_ornim(channels, os.path.join(out, "2_ornim_cfi_vs_time.png"), f"{name}: ORNIM CFI vs time")
        plot_overlay(c, overlay_i, channels, os.path.join(out, "3_overlay_comind_ornim.png"),
                     f"{name}: CoMind vs ORNIM")
    plot_comind_heatmap(c, os.path.join(out, "extra_comind_heatmap.png"), f"{name}: CoMind rBFI across ToFs")
    plot_comind_quality(c, os.path.join(out, "extra_comind_fit_quality.png"), f"{name}: CoMind fit quality")

    print("\n" + "=" * 70)
    print(f"PNG files now in {out}:")
    for f in sorted(os.listdir(out)):
        if f.endswith(".png"):
            print("   ", f)
    print("=" * 70)


if __name__ == "__main__":
    main()
