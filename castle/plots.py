"""Final plots. Run final_data.py first; this only draws results/final.json. Writes plots/*.png.

Regimes are simulated fields, not observed entries: the leaderboard publishes scores
only. Figure 1 and the real-entry marks use real data; the rest is simulation.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

from attack import arms_race
from field import random_allocs, scenario
from search import low_castles
from stress2 import REAL, V5, conc, sharp_g1, sharp_g2

OUT = Path("plots")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#6250d6")
PALE = "#a9c9ef"
OTHERS = (ORANGE, VIOLET, GREEN, MAGENTA)
FAIR = {"g1": 7.5, "g2": 3.75}
NAME = {"g1": "Game 1", "g2": "Game 2"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def combo(a):
    return ",".join(str(v) for v in a)


def load():
    d = json.load(open("results/final.json"))
    for g in ("g1", "g2"):
        strategies = d[f"{g}_strategies"]
        d[g]["keys"] = list(strategies)
        d[g]["allocs"] = list(strategies.values())
        d[g]["labels"] = [f"{k.strip()}  ·  {combo(v)}" for k, v in strategies.items()]
        d[g]["final"] = next(i for i, k in enumerate(strategies) if "Final" in k)
        d[g]["regimes"] = [d["labels"].get(n, n) for n in d[g]["names"]]
        d[g]["order"] = np.argsort(d[g]["curve"]).tolist()
    d["g1"]["s"] = np.array(d["g1"]["scores"])
    d["g2"]["p"] = np.array(d["g2"]["places"])
    d["g2"]["s"] = d["g2"]["p"][:, :, 0]
    return d


def title(fig, text, sub):
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.14 / h, text, fontsize=16, fontweight="bold", va="top")
    fig.text(0.015, 1 - 0.47 / h, sub, fontsize=11, color=MUTED, va="top")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def heat(ax, grid, g, cmap, norm, fmt, top_labels=True, light_above=None):
    rows, cols = grid.shape
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-.5, cols - .5)
    ax.set_yticks(range(rows), g["labels"], fontsize=10.5, color=INK)
    ax.get_yticklabels()[g["final"]].set_fontweight("bold")
    ax.set_xticks(range(cols), [g["regimes"][j] for j in g["order"]] if top_labels else [""] * cols,
                  rotation=38, ha="left", rotation_mode="anchor", fontsize=10.5)
    ax.tick_params(length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks(np.arange(-.5, cols), minor=True)
    ax.set_yticks(np.arange(-.5, rows), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(rows):
        for j in range(cols):
            light = light_above is not None and grid[i, j] >= light_above
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=10,
                    color=SURFACE if light else INK)
    ax.add_patch(Rectangle((-.5, g["final"] - .5), cols, 1, fill=False, ec=INK, lw=2, clip_on=False, zorder=5))


def fig_real(d):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(top=0.8, bottom=0.11, left=0.05, right=0.985, wspace=0.13)
    title(fig, "Where the final solutions sit against the real leaderboard",
          "Bars are real published scores. Solid lines are real entries; dashed lines are simulated estimates for the finals.")
    marks = {"g1": [(8.0, "1,26,29,43,1", ORANGE), (8.1, "0,0,37,6,57", GREEN)], "g2": [(6.07, "9,36,55,0,0", ORANGE)]}
    for ax, g in zip(axes, ("g1", "g2")):
        p = np.array(d["pub"][g])
        ax.hist(p, bins=np.arange(0, p.max() + .1, .1), color=PALE, edgecolor=SURFACE, linewidth=.8)
        i, w = d[g]["final"], np.array(d[g]["weights"])
        est = float(w @ d[g]["s"][:, i])
        ax.axvline(est, color=BLUE, lw=2.4, ls="--", label=f"Final estimate {est:.2f}  ·  {combo(d[g]['allocs'][i])}")
        for x, alloc, colour in marks[g]:
            ax.axvline(x, color=colour, lw=2, label=f"Real entry {x}  ·  {alloc}")
        ax.axvline(FAIR[g], color=MUTED, ls=":", lw=1.6, label=f"Fair share {FAIR[g]}")
        ax.legend(frameon=False, loc="upper left", fontsize=10.5)
        ax.set_title(f"{NAME[g]}: {len(p):,} real entries, top score {p.max():.2f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Entries")
        ax.grid(axis="x", visible=False)
    axes[0].set_xlim(4, 9.6)
    axes[1].set_xlim(0.5, 7)
    save(fig, "1_real_leaderboard.png")


def fig_shapes(rng):
    groups = {
        "Riddler entries (real, mapped)": REAL, "Focal splits": scenario(rng, 3000),
        "Heavy on 1–2 castles": conc(rng)(3000), "Low-castle players": low_castles(rng, 3000),
        "Castle-5 race": arms_race(rng, 3000), "Pure equilibrium, Game 1": sharp_g1(rng, 3000),
        "Pure equilibrium, Game 2": sharp_g2(rng, 3000), "Random entries": random_allocs(3000, rng),
    }
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.05, right=0.985, hspace=0.3, wspace=0.07)
    title(fig, "What each contestant regime looks like",
          "Bars show the mean soldiers per castle. Whiskers span the 10th to 90th percentile of entries.")
    for ax, (name, f) in zip(axes.flat, groups.items()):
        mean = f.mean(0)
        lo, hi = np.quantile(f, [0.1, 0.9], axis=0)
        ax.bar(V5, mean, width=0.6, color=BLUE)
        ax.errorbar(V5, mean, yerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], fmt="none",
                    ecolor=INK, elinewidth=1.2, capsize=3)
        ax.set_title(name, fontsize=11.5)
        ax.set_xticks(V5)
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Castle (value in points)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Soldiers")
    save(fig, "2_contestant_regimes.png")


def fig_matrix(d, g, name, low, high, heading):
    data = d[g]
    grid = data["s"].T[:, data["order"]]
    fig, ax = plt.subplots(figsize=(19, 9.2))
    fig.subplots_adjust(top=0.72, bottom=0.015, left=0.185, right=0.94)
    title(fig, heading, f"{NAME[g]} score by regime; fair share {FAIR[g]}. Blue beats fair, orange falls short. "
                        "Regimes run left to right from best to worst fit with the real leaderboard.")
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ec", BLUE])
    heat(ax, grid, data, cmap, TwoSlopeNorm(FAIR[g], low, high), "{:.1f}")
    save(fig, name)


def fig_sharpness(d):
    x = np.array(d["sharpness"]["shares"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.05, right=0.9, wspace=0.34)
    title(fig, "How each solution holds up as the field gets sharper",
          "Riddler entries mixed with near-equilibrium play, averaged over 5 draws. Dashed lines mark the real leaderboard.")
    for ax, g in zip(axes, ("g1", "g2")):
        data = d[g]
        y = np.array(d["sharpness"][g])
        keep = [i for i, k in enumerate(data["keys"]) if k.startswith("Solution")]
        keep.append(data["keys"].index("Proportional"))
        rest = iter(OTHERS)
        colours = {i: BLUE if i == data["final"] else AQUA if i == keep[-1] else next(rest) for i in keep}
        for i in keep:
            ax.plot(x, y[:, i], color=colours[i], lw=3 if i == data["final"] else 1.8, label=data["labels"][i],
                    zorder=3 if i == data["final"] else 2)
        gap = (y[:, keep].max() - y[:, keep].min()) * 0.052
        last = -99
        for i in sorted(keep, key=lambda i: y[-1, i]):
            last = max(y[-1, i], last + gap)
            ax.text(91, last, combo(data["allocs"][i]), va="center", fontsize=10,
                    fontweight="bold" if i == data["final"] else "normal")
        p = np.array(d["pub"][g])
        for value, label in ((p.max(), "Real top"), (np.quantile(p, .9), "Real 90th percentile")):
            ax.axhline(value, color=MUTED, ls="--", lw=1)
            ax.text(89, value, label, va="bottom", ha="right", color=MUTED, fontsize=10)
        ax.axhline(FAIR[g], color=MUTED, ls=":", lw=1.2)
        ax.set_title(NAME[g])
        ax.set_xlim(0, 90)
        ax.set_xlabel("Near-equilibrium share of the field (%)")
        ax.set_ylabel("Score")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
    save(fig, "5_score_vs_sharpness.png")


def fig_finish(d):
    data = d["g2"]
    best = data["names"].index("20% sharp")
    shares = data["p"][best, :, 1:5]
    order = np.argsort(shares @ np.array([9, 4, 2, 0]))[::-1]
    fig, ax = plt.subplots(figsize=(14, 8.2))
    fig.subplots_adjust(top=0.82, bottom=0.07, left=0.3, right=0.98)
    title(fig, "Game 2 finishing places in the best-fit field",
          "Riddler entries plus 20% near-equilibrium play. Ties are shared between tied places. Score in brackets; sorted by score.")
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for k, (label, colour) in enumerate(zip(("1st (pays 9)", "2nd (pays 4)", "3rd (pays 2)", "4th (pays 0)"),
                                            (BLUE, ORANGE, AQUA, YELLOW))):
        v = shares[order, k]
        ax.barh(y, v, left=left, height=0.66, color=colour, edgecolor=SURFACE, linewidth=2, label=label)
        for yy, a, b in zip(y, left, v):
            if b >= 0.05:
                ax.text(a + b / 2, yy, f"{b:.0%}", ha="center", va="center", fontsize=10.5)
        left += v
    ax.set_yticks(y, [f"{data['labels'][i]}   ({data['s'][best, i]:.2f})" for i in order], fontsize=10.5, color=INK)
    ax.get_yticklabels()[list(order).index(data["final"])].set_fontweight("bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10.5)
    save(fig, "6_game2_finishing_places.png")


def fig_place_odds(d):
    data = d["g2"]
    fig, axes = plt.subplots(2, 1, figsize=(19, 15.5), gridspec_kw={"hspace": 0.04})
    fig.subplots_adjust(top=0.83, bottom=0.01, left=0.2, right=0.94)
    title(fig, "Game 2 odds of finishing first and last, in every regime",
          "Regimes run left to right from best to worst fit with the real leaderboard. Ties are shared between tied places.")
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", BLUE])
    panels = ((1, "Finish 1st (darker is better)", 1.0, 0.72), (4, "Finish last (darker is worse)", 0.55, 0.42))
    for ax, (col, label, vmax, light) in zip(axes, panels):
        grid = data["p"][:, :, col].T[:, data["order"]]
        heat(ax, grid, data, cmap, Normalize(0, vmax), "{:.0%}", top_labels=ax is axes[0], light_above=light)
        ax.set_ylabel(label, fontsize=12, color=INK, fontweight="bold", labelpad=10)
    save(fig, "7_game2_place_odds_by_regime.png")


def fig_spread(d):
    fig, axes = plt.subplots(2, 1, figsize=(15, 12.5))
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.27, right=0.975, hspace=0.27)
    title(fig, "Every strategy's range of scores across all regimes",
          "Each dot is one regime. The black bar and number show the average weighted by fit to the real data. Sorted by that average.")
    for ax, g in zip(axes, ("g1", "g2")):
        data = d[g]
        s, w = data["s"], np.array(data["weights"])
        avg = w @ s
        order = np.argsort(-avg)
        y = np.arange(len(order))[::-1]
        rng = np.random.default_rng(1)
        for yy, i in zip(y, order):
            ax.scatter(s[:, i], yy + rng.uniform(-.2, .2, len(s)), s=38, color=BLUE if i == data["final"] else PALE,
                       edgecolor=SURFACE, linewidth=.8, zorder=3)
            ax.plot([avg[i]] * 2, [yy - .36, yy + .36], color=INK, lw=2.6, zorder=4)
            ax.text(s[:, i].max() + .08, yy, f"{avg[i]:.2f}", va="center", fontsize=10)
        ax.axvline(FAIR[g], color=MUTED, ls="--", lw=1.2)
        ax.text(FAIR[g], len(order) - .3, f"Fair share {FAIR[g]}", ha="center", color=MUTED, fontsize=10)
        ax.set_yticks(y, [data["labels"][i] for i in order], fontsize=10.5, color=INK)
        ax.get_yticklabels()[list(order).index(data["final"])].set_fontweight("bold")
        ax.set_title(NAME[g], pad=18)
        ax.set_xlabel("Score")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
    save(fig, "8_score_range_by_strategy.png")


def fig_evidence(d):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.8))
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.18, right=0.985, wspace=0.42)
    title(fig, "Which regimes predict the real entries?",
          "Dots show each regime's predicted score; solid lines show the real score. Regimes sorted by fit to the score curve.")
    panels = (("g1", [("Real entry (8.0)", 8.0, BLUE), ("Real entry (8.1)", 8.1, ORANGE)]),
              ("g2", [("Real entry (6.07)", 6.07, BLUE)]))
    for ax, (g, entries) in zip(axes, panels):
        data = d[g]
        order = data["order"]
        y = np.arange(len(order))[::-1]
        for key, actual, colour in entries:
            i = data["keys"].index(key)
            ax.scatter(data["s"][order, i], y, s=64, color=colour, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=f"{combo(data['allocs'][i])}, real {actual}")
            ax.axvline(actual, color=colour, lw=1.6, zorder=2)
        ax.set_yticks(y, [data["regimes"][j] for j in order], fontsize=10, color=INK)
        ax.set_title(NAME[g])
        ax.set_xlabel("Predicted score")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=10)
    save(fig, "9_regimes_vs_real_entries.png")


def fig_win_curves(d, rng):
    pick = d["g1"]["allocs"][d["g1"]["final"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.06, right=0.985, wspace=0.06)
    title(fig, f"Why the Game 1 final sends {combo(pick)}",
          "Chance of winning each castle at each bid, ties counted as half. Diamonds mark the final solution's bids.")
    colours = (YELLOW, MAGENTA, AQUA, ORANGE, BLUE)
    low = np.vstack([REAL[rng.integers(len(REAL), size=1800)], low_castles(rng, 1200)])
    fields = (("Riddler entries", REAL), ("40% low-castle players", low), ("Pure equilibrium", sharp_g1(rng, 3000)))
    for ax, (name, f) in zip(axes, fields):
        for c in range(5):
            pmf = np.bincount(f[:, c], minlength=101)[:101] / len(f)
            below = np.concatenate([[0.0], np.cumsum(pmf)[:-1]])
            line = (below + 0.5 * pmf)[:71]
            ax.plot(range(71), line, color=colours[c], lw=2, label=f"Castle {c + 1} ({c + 1} {'pt' if c == 0 else 'pts'})")
            ax.scatter(pick[c], line[pick[c]], marker="D", s=62, color=colours[c], edgecolor=INK, linewidth=1, zorder=4)
        ax.set_title(name)
        ax.set_xlim(0, 70)
        ax.set_xlabel("Soldiers sent")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Chance of winning that castle")
    axes[1].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=10)
    save(fig, "10_game1_castle_win_curves.png")


def main():
    d = load()
    pub = json.load(open("../data/leaderboard_latest.json"))
    d["pub"] = {g: [r[1] for r in pub[g]] for g in ("g1", "g2")}
    rng = np.random.default_rng(15)
    fig_real(d)
    fig_shapes(rng)
    fig_matrix(d, "g1", "3_game1_scores_by_regime.png", 3, 12,
               "Game 1: every strategy against every contestant regime")
    fig_matrix(d, "g2", "4_game2_scores_by_regime.png", 0, 9,
               "Game 2: every strategy against every contestant regime")
    fig_sharpness(d)
    fig_finish(d)
    fig_place_odds(d)
    fig_spread(d)
    fig_evidence(d)
    fig_win_curves(d, rng)


if __name__ == "__main__":
    main()
