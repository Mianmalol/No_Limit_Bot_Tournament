"""Plots of contestant regimes and how strategies score against them. Writes plots/*.png.

Regimes are simulated fields, not observed entries: the leaderboard publishes scores
only. Real score curves appear only in the figures that say so.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from field import random_allocs, scenario
from riddler import scores
from search import run
from stress2 import EXE, REAL, V5, conc, mix, sharp_g1, sharp_g2

OUT = Path("plots")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SIZE = 3000
TABLES = 40000

STRATEGIES = {
    "Game 1 pick\n3,5,4,43,45": (3, 5, 4, 43, 45),
    "Game 2 pick\n0,39,0,61,0": (0, 39, 0, 61, 0),
    "Earlier Game 1\n6,16,1,36,41": (6, 16, 1, 36, 41),
    "Earlier Game 2\n13,37,48,0,2": (13, 37, 48, 0, 2),
    "Oldest Game 2\n6,31,0,63,0": (6, 31, 0, 63, 0),
    "All on 1 & 5\n28,0,0,0,72": (28, 0, 0, 0, 72),
    "Castles 3 & 5\n0,0,39,0,61": (0, 0, 39, 0, 61),
    "Top two\n0,0,0,40,60": (0, 0, 0, 40, 60),
    "Top three\n0,0,33,33,34": (0, 0, 33, 33, 34),
    "Proportional\n7,13,20,27,33": (7, 13, 20, 27, 33),
    "Round steps\n10,15,20,25,30": (10, 15, 20, 25, 30),
    "Even split\n20,20,20,20,20": (20, 20, 20, 20, 20),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def regimes(rng, sharp):
    return {
        "Riddler\nentries": REAL[rng.integers(len(REAL), size=SIZE)],
        "Focal\nsplits": scenario(rng, SIZE),
        "Heavy on\n1–2 castles": conc(rng)(SIZE),
        "Near-\nequilibrium": sharp(rng, SIZE),
        "Random": random_allocs(SIZE, rng),
        "Riddler +\n20% sharp": mix(rng, [(0.2, lambda n: sharp(rng, n))]),
    }


def g2_eval(field, x, d):
    np.savetxt(f"{d}/f", field, fmt="%d")
    np.savetxt(f"{d}/c", x, fmt="%d")
    out = subprocess.run([EXE, "eval", f"{d}/f", f"{d}/c", str(TABLES), "5"], capture_output=True,
                         text=True, check=True).stdout
    return np.array([float(line.split()[0]) for line in out.splitlines()])


def title(fig, text, sub):
    fig.text(0.02, 0.985, text, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.985 - 0.32 / fig.get_figheight(), sub, fontsize=10.5, color=MUTED, va="top")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def fig_real_scores(p1, p2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.06, right=0.98, wspace=0.16)
    title(fig, "Real scores show two camps in Game 1 and one big shared entry in Game 2",
          "Published leaderboard, Nov 1, 12:36 AM PT. Scores only: no allocations are public.")
    for ax, p, step, fair, name in ((axes[0], p1, 0.1, 7.5, "Game 1"), (axes[1], p2, 0.1, 3.75, "Game 2")):
        ax.hist(p, bins=np.arange(p.min() // 1, p.max() + step, step), color=BLUE, edgecolor=SURFACE,
                linewidth=1)
        ax.axvline(fair, color=MUTED, ls="--", lw=1.2)
        ax.set_title(f"{name}: {len(p):,} entries, top {p.max():.2f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Entries")
        ax.grid(axis="x", visible=False)
    axes[0].set_xlim(3.5, 9.5)
    top = axes[0].get_ylim()[1]
    axes[0].text(7.45, top * 0.93, "Fair share 7.5", ha="right", color=MUTED)
    axes[0].annotate(f"Top 1% sits\nwithin {p1.max() - np.quantile(p1, .99):.2f}", (9.1, 42), (9.05, top * 0.74),
                     color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED))
    top = axes[1].get_ylim()[1]
    axes[1].text(3.7, top * 0.93, "Fair share 3.75", ha="right", color=MUTED)
    n = int((p2 == 4.847).sum())
    axes[1].annotate(f"{n} entries tied\nat exactly 4.847", (4.9, top * 0.8), (5.25, top * 0.78), color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED))
    save(fig, "1_real_scores.png")


def fig_regime_shapes(rng):
    groups = {
        "Riddler entries (real, mapped)": REAL,
        "Focal splits (assumed)": scenario(rng, SIZE),
        "Heavy on 1–2 castles (assumed)": conc(rng)(SIZE),
        "Near-equilibrium, Game 1": sharp_g1(rng, SIZE),
        "Near-equilibrium, Game 2": sharp_g2(rng, SIZE),
        "Random": random_allocs(SIZE, rng),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.06, right=0.98, hspace=0.34, wspace=0.08)
    title(fig, "Most regimes share a similar average; the spread is what separates them",
          "Bars: mean soldiers per castle. Whiskers: 10th to 90th percentile of entries.")
    for ax, (name, f) in zip(axes.flat, groups.items()):
        mean = f.mean(0)
        lo, hi = np.quantile(f, [0.1, 0.9], axis=0)
        ax.bar(V5, mean, width=0.55, color=BLUE)
        ax.errorbar(V5, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], fmt="none",
                    ecolor=INK, elinewidth=1.3, capsize=4)
        ax.set_title(name)
        ax.set_xticks(V5)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Castle (also its points)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Soldiers")
    save(fig, "2_regime_shapes.png")


def heatmap(ax, grid, cols, centre, low, high, unit):
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ec", BLUE])
    ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(centre, low, high), aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=10)
    ax.set_yticks(range(len(STRATEGIES)), list(STRATEGIES), fontsize=9)
    ax.tick_params(length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-.5, grid.shape[1] - .5)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color=INK, fontsize=10.5)
    ax.set_xticks(np.arange(-.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)


def fig_matrix(g1, g2, cols1, cols2):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
    fig.subplots_adjust(top=0.8, bottom=0.03, left=0.095, right=0.99, wspace=0.42)
    title(fig, "No strategy wins every regime, and Game 2 punishes spreading out far harder",
          "Mean score against each simulated regime. Blue beats a fair share, orange falls short.")
    heatmap(axes[0], g1, cols1, 7.5, 3, 12, "points")
    heatmap(axes[1], g2, cols2, 3.75, 0, 9, "payout")
    axes[0].text(-0.5, -1.45, "Game 1, out of 15 (fair share 7.5)", fontweight="bold")
    axes[1].text(-0.5, -1.45, "Game 2, out of 9 (fair share 3.75)", fontweight="bold")
    save(fig, "3_strategy_by_regime.png")


def fig_sharpness(rng, p1, p2, d):
    shares = np.arange(0, 1.0, 0.1)
    lines1 = {"Game 1 pick": (3, 5, 4, 43, 45), "Earlier Game 1": (6, 16, 1, 36, 41),
              "Proportional": (7, 13, 20, 27, 33)}
    lines2 = {"Game 2 pick": (0, 39, 0, 61, 0), "Earlier Game 2": (13, 37, 48, 0, 2),
              "Top two": (0, 0, 0, 40, 60)}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.06, right=0.89, wspace=0.46)
    title(fig, "As the field sharpens, every edge shrinks toward a fair share",
          "Riddler entries mixed with near-equilibrium play, averaged over 5 draws. Dashed lines are the real leaderboard;\n"
          "Game 1 stays well below the real top, so this mix fits Game 1 poorly.")
    for ax, lines, sharp, pub, name, fair in ((axes[0], lines1, sharp_g1, p1, "Game 1", 7.5),
                                              (axes[1], lines2, sharp_g2, p2, "Game 2", 3.75)):
        x = np.array(list(lines.values()))
        rows = []
        for s in shares:
            draws = []
            for _ in range(5):
                f = mix(rng, [(float(s), lambda n: sharp(rng, n))])
                draws.append(scores(x, f, V5) if name == "Game 1" else g2_eval(f, x, d))
            rows.append(np.mean(draws, 0))
        rows = np.array(rows)
        gap = 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * (rows.max() - rows.min())
        placed = {}
        for i in np.argsort(rows[-1]):
            placed[i] = max(rows[-1, i], max(placed.values(), default=-9) + gap)
        for i, (label, colour) in enumerate(zip(lines, (BLUE, ORANGE, AQUA))):
            ax.plot(shares * 100, rows[:, i], color=colour, lw=2, label=label)
            ax.text(91, placed[i], label, va="center", color=INK, fontsize=10)
        for value, label in ((pub.max(), "Real top"), (np.quantile(pub, .9), "Real 90th pct")):
            ax.axhline(value, color=MUTED, ls="--", lw=1)
            ax.text(1, value, label, va="bottom", color=MUTED, fontsize=10)
        ax.axhline(fair, color=GRID, lw=2)
        ax.set_title(name)
        ax.set_xlim(0, 90)
        ax.set_xlabel("Near-equilibrium share of field (%)")
        ax.set_ylabel("Score")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1, loc="upper right",
                  fontsize=9.5)
    save(fig, "4_score_vs_sharpness.png")


def fig_finish(rng, d):
    names = ["Game 2 pick\n0,39,0,61,0", "Earlier Game 2\n13,37,48,0,2", "Oldest Game 2\n6,31,0,63,0",
             "All on 1 & 5\n28,0,0,0,72", "Top two\n0,0,0,40,60", "Proportional\n7,13,20,27,33",
             "Even split\n20,20,20,20,20"]
    field = mix(rng, [(0.2, lambda n: sharp_g2(rng, n))])
    x = np.array([STRATEGIES[n] for n in names])
    lines = run("places", field, x, 200000, 9, d)
    shares = np.array([[float(v) for v in line.split()[1:5]] for line in lines])
    fig, ax = plt.subplots(figsize=(11.5, 6))
    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.17, right=0.97)
    title(fig, "Game 2 pays for first place, so concentrated entries beat steady ones",
          "Share of tables by finish in the best-fit field (Riddler + 20% sharp). Ties are shared between the tied places.")
    left = np.zeros(len(names))
    y = np.arange(len(names))[::-1]
    for k, (label, colour) in enumerate(zip(("1st (9)", "2nd (4)", "3rd (2)", "4th (0)"),
                                            (BLUE, ORANGE, AQUA, YELLOW))):
        ax.barh(y, shares[:, k], left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2,
                label=label)
        for i, v in enumerate(shares[:, k]):
            if v >= 0.07:
                ax.text(left[i] + v / 2, y[i], f"{v:.0%}", ha="center", va="center", color=INK, fontsize=10)
        left += shares[:, k]
    ax.set_yticks(y, names, fontsize=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
    save(fig, "5_game2_finish_places.png")


def fig_place_odds():
    d = json.load(open("results/search/g2_final.json"))
    names, r = d["names"], np.array(d["results"])
    labels = {(0, 39, 0, 61, 0): "Game 2 pick\n0,39,0,61,0", (13, 37, 48, 0, 2): "Earlier\n13,37,48,0,2",
              (6, 31, 0, 63, 0): "Oldest\n6,31,0,63,0", (28, 0, 0, 0, 72): "1 & 5\n28,0,0,0,72"}
    idx = [d["finalists"].index(list(k)) for k in labels]
    err = np.array([d["errors"][n] for n in names])
    order = np.argsort(err)
    fig, axes = plt.subplots(1, 3, figsize=(15, 10.5), sharey=True)
    fig.subplots_adjust(top=0.83, bottom=0.02, left=0.13, right=0.99, wspace=0.05)
    title(fig, "Every strategy has regimes that break it, and no column wins every row",
          "Game 2 score and exact place odds on fresh fields. Rows run from best to worst fit with the real leaderboard.")
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", BLUE])
    for ax, col, name, fmt, vmax in ((axes[0], 0, "Score (fair 3.75)", "{:.1f}", 9), (axes[1], 1, "Finish 1st", "{:.0%}", 1),
                                     (axes[2], 4, "Finish last", "{:.0%}", 0.5)):
        grid = r[order][:, idx, col]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(idx)), list(labels.values()), fontsize=9)
        ax.tick_params(length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_yticks(range(len(names)), [names[i] for i in order], fontsize=10)
        ax.grid(False)
        ax.set_xlim(-.5, len(idx) - .5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                dark = grid[i, j] / vmax > 0.62
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=10,
                        color=SURFACE if dark else INK)
        ax.text(-0.5, -1.55, name, fontweight="bold", fontsize=12)
    save(fig, "7_place_odds_by_regime.png")


def fig_spread():
    d = json.load(open("results/search/g2_final.json"))
    names, r = d["names"], np.array(d["results"])
    w = np.exp(-np.array([d["errors"][n] for n in names]) / 0.15)
    w /= w.sum()
    show = [((0, 39, 0, 61, 0), "Game 2 pick 0,39,0,61,0"), ((28, 0, 0, 0, 72), "All on 1 & 5  28,0,0,0,72"),
            ((6, 31, 0, 63, 0), "Oldest 6,31,0,63,0"), ((13, 37, 48, 0, 2), "Earlier 13,37,48,0,2"),
            ((10, 4, 6, 0, 80), "Max worst case 10,4,6,0,80")]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.22, right=0.97)
    title(fig, "The Game 2 pick has the best average, with no drop much below a fair share",
          "Game 2 score in each of 21 regimes (dots). Bars mark the leaderboard-weighted average.")
    y = np.arange(len(show))[::-1]
    jitter_rng = np.random.default_rng(0)
    for k, (alloc, _) in enumerate(show):
        s = r[:, d["finalists"].index(list(alloc)), 0]
        ax.scatter(s, y[k] + jitter_rng.uniform(-.14, .14, len(s)), s=46, color=BLUE, alpha=.65,
                   edgecolor=SURFACE, linewidth=1, zorder=3, label="One regime" if k == 0 else None)
        ax.plot([s @ w] * 2, [y[k] - .3, y[k] + .3], color=INK, lw=2.5, zorder=4,
                label="Weighted average" if k == 0 else None)
    ax.axvline(3.75, color=MUTED, ls="--", lw=1.2)
    ax.text(3.72, -.42, "Fair share 3.75", ha="right", color=MUTED, fontsize=10)
    ax.set_yticks(y, [n for _, n in show])
    ax.set_xlabel("Game 2 score")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10)
    save(fig, "8_score_spread_across_regimes.png")


def fig_win_curves(rng):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.07, right=0.98, wspace=0.08)
    title(fig, "Real entries leave steep win-rate jumps at popular bids; equilibrium play smooths them away",
          "Game 1 chance of beating a random opponent at each bid, ties counted as half. Numbers mark each castle.")
    colours = (YELLOW, "#e87ba4", AQUA, ORANGE, BLUE)
    for ax, f, name in ((axes[0], REAL, "Riddler entries"), (axes[1], sharp_g1(rng, SIZE), "Near-equilibrium")):
        for c in range(5):
            pmf = np.bincount(f[:, c], minlength=101)[:101] / len(f)
            below = np.concatenate([[0.0], np.cumsum(pmf)[:-1]])
            ax.plot(range(71), (below + 0.5 * pmf)[:71], color=colours[c], lw=2, label=f"Castle {c + 1}")
            curve = (below + 0.5 * pmf)[:71]
            at = int(np.argmax(curve >= 0.85))
            ax.text(at, 0.85, str(c + 1), ha="center", va="center", fontsize=10, color=INK,
                    bbox=dict(boxstyle="circle,pad=0.15", fc=SURFACE, ec=colours[c], lw=1.5))
        ax.set_title(name)
        ax.set_xlim(0, 70)
        ax.set_xlabel("Soldiers sent")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Win rate at that castle")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9.5)
    save(fig, "6_castle_win_curves.png")


def main():
    pub = json.load(open("../data/leaderboard_scores.json"))
    p1 = np.array([r[1] for r in pub["g1"]])
    p2 = np.array([r[1] for r in pub["g2"]])
    rng = np.random.default_rng(12)
    x = np.array(list(STRATEGIES.values()))
    fig_real_scores(p1, p2)
    fig_regime_shapes(rng)
    r1, r2 = regimes(rng, sharp_g1), regimes(rng, sharp_g2)
    g1 = np.array([scores(x, f, V5) for f in r1.values()]).T
    with tempfile.TemporaryDirectory() as d:
        g2 = np.array([g2_eval(f, x, d) for f in r2.values()]).T
        fig_matrix(g1, g2, list(r1), list(r2))
        fig_sharpness(rng, p1, p2, d)
        fig_finish(rng, d)
    fig_win_curves(rng)
    fig_place_odds()
    fig_spread()
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(g1)
    print(g2)


if __name__ == "__main__":
    main()
