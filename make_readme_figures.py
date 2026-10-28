"""
Figures for the summary at the top of README.md, from the camera data in this
repo (camera_minutes.csv.gz, svalbard_minutes.csv.gz) and the deployed model.
Writes figures/<name>_light.png and figures/<name>_dark.png. Takes a few minutes.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = Path("figures")
DELTA = 7
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e6e5e1", ref="#a3a29c",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#33332f", ref="#6f6e68",
                 blue="#3987e5", orange="#d95926", aqua="#199e70"),
}
NAMES = {"chibougamau": "Chibougamau", "kapuskasing": "Kapuskasing", "athabasca": "Athabasca", "the pas": "The Pas",
         "gakona": "Gakona", "whitehorse": "Whitehorse", "kiruna": "Kiruna", "skibotn": "Skibotn", "tromso": "Tromsø",
         "kho svalbard": "Svalbard"}


def load():
    bundle = joblib.load("aurora_ebm_model_final_deployment.joblib")
    model, feats = bundle["pipeline"], bundle["featureslow"]
    df = pd.concat([pd.read_csv("camera_minutes.csv.gz"), pd.read_csv("svalbard_minutes.csv.gz")], ignore_index=True)
    df["abs_mlat"] = df["mlat"].abs()
    df["cloud_cover"] = 0.0
    for d in [0, DELTA]:
        x = df.copy()
        x["mlat"] = np.sign(x["mlat"]) * (x["abs_mlat"] - d)
        x["oval_dist"] = x["mlat"].abs() - (66.0 - 2.0 * x["kp"])
        df[f"p{d}"] = model.predict_proba(x[feats])[:, 1]
    rows = []
    for c, g in df.groupby("camera"):
        sure = g[g[f"p{DELTA}"] > 0.7]
        rows.append({"camera": c, "mlat": g["abs_mlat"].mean(), "rate": g["aurora"].mean(),
                     "auc": roc_auc_score(g["aurora"], g["p0"]), "m1_at": g["p0"].mean(),
                     "m1_south": g[f"p{DELTA}"].mean(), "n_sure": len(sure),
                     "precision": sure["aurora"].mean() if len(sure) else np.nan})
    summary = pd.DataFrame(rows).sort_values("mlat").reset_index(drop=True)

    # mismatch between the model's average and the camera's aurora rate, per camera and Kp bin, for each shift
    cams = df[df["camera"] != "kho svalbard"]
    sample = pd.concat([g.sample(min(len(g), 40000), random_state=0) for _, g in cams.groupby("camera")])
    kp_bin = pd.cut(sample["kp"], [-0.1, 1.5, 3, 4.5, 9.1])
    curve = {}
    for d in range(13):
        x = sample.copy()
        x["mlat"] = np.sign(x["mlat"]) * (x["abs_mlat"] - d)
        x["oval_dist"] = x["mlat"].abs() - (66.0 - 2.0 * x["kp"])
        x["p"] = model.predict_proba(x[feats])[:, 1]
        g = x.groupby([x["camera"], kp_bin], observed=True).agg(p=("p", "mean"), rate=("aurora", "mean"))
        curve[d] = (g["p"] - g["rate"]).abs().mean()
    return summary, pd.Series(curve)


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, ax, t, title, subtitle, name, mode):
    fig.patch.set_facecolor(t["surface"])
    fig.text(0.02, 0.97, title, color=t["text"], fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.91, subtitle, color=t["muted"], fontsize=9, va="top")
    fig.savefig(OUT / f"{name}_{mode}.png", dpi=160, facecolor=t["surface"])
    plt.close(fig)


def fig_auc(s, t, mode):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.12)
    style(ax, t)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(s))[::-1]
    colors = [t["ref"] if c == "kho svalbard" else t["blue"] for c in s["camera"]]
    ax.barh(y, s["auc"] - 0.5, left=0.5, height=0.6, color=colors)
    for yi, (a, m) in zip(y, zip(s["auc"], s["mlat"])):
        ax.text(a + 0.006, yi, f"{a:.2f}", va="center", fontsize=9, color=t["text"])
    ax.set_yticks(y, [f"{NAMES[c]} ({m:.0f}°)" for c, m in zip(s["camera"], s["mlat"])], color=t["muted"])
    ax.axvline(0.83, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.832, y.max() + 0.75, "on citizen reports (0.83)", fontsize=8, color=t["muted"])
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("AUC within each camera (0.5 = coin flip)", color=t["muted"], fontsize=9)
    finish(fig, ax, t, "Ranking holds on minutes nobody chose to report",
           "Deployed citizen model on clear, dark camera minutes. Svalbard (polar cap) is the exception.",
           "auc_by_camera", mode)


def fig_sketch(t, mode):
    """Not to scale: heights are stretched so the geometry is visible."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.04)
    ax.set_facecolor(t["surface"])
    ax.axis("off")
    ax.set_xlim(-130, 1010)
    ax.set_ylim(-80, 190)
    ax.plot([-130, 1010], [0, 0], color=t["muted"], linewidth=1.5)
    # aurora photo: BLM Alaska, public domain
    ax.imshow(plt.imread("figures/aurora_patch.jpg"), extent=[770, 830, 108, 150], aspect="auto", zorder=2)
    ax.text(845, 129, "aurora\n~100-110 km up", color=t["text"], fontsize=9, va="center")
    # camera looking straight up
    ax.plot([800, 800], [0, 102], color=t["blue"], linewidth=1.5, linestyle=(0, (3, 3)))
    ax.scatter([800], [0], s=90, marker="s", color=t["blue"], zorder=3)
    ax.text(800, -14, "camera\nsees it overhead", color=t["text"], fontsize=9, ha="center", va="top")
    # person looking low to the north
    ax.plot([0, 775], [0, 108], color=t["orange"], linewidth=1.5, linestyle=(0, (3, 3)))
    ax.scatter([0], [0], s=90, marker="o", color=t["orange"], zorder=3)
    ax.text(0, -14, "human observer", color=t["text"], fontsize=9, ha="center", va="top")
    ax.text(120, 6, "~4-5°", color=t["muted"], fontsize=9)
    ax.annotate("", xy=(800, -52), xytext=(0, -52), color=t["muted"],
                arrowprops=dict(arrowstyle="<->", color=t["muted"], linewidth=1))
    ax.text(400, -58, "~800 km = 7° of magnetic latitude", color=t["muted"], fontsize=9, ha="center", va="top")
    finish(fig, ax, t, "A person sees aurora that is overhead about 7° further north",
           "So the citizen model for an observer should match a camera ~7° north. Not to scale.",
           "horizon_sketch", mode)


def fig_shift_error(curve, t, mode):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.76, bottom=0.15)
    style(ax, t)
    ax.plot(curve.index, curve.values, color=t["blue"], linewidth=2)
    ax.scatter(curve.index, curve.values, s=30, color=t["blue"], edgecolor=t["surface"], linewidth=1.5, zorder=3)
    best = curve.idxmin()
    ax.annotate(f"best near {best}°", xy=(best, curve[best]), xytext=(best, curve[best] + 0.08), ha="center",
                color=t["text"], fontsize=9, arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=1))
    ax.set_xticks(range(0, 13))
    ax.set_ylim(0, max(0.4, curve.max() + 0.05))
    ax.set_xlabel("How far north the camera is, compared with the person (° magnetic latitude)", color=t["muted"],
                  fontsize=9)
    ax.set_ylabel("Mismatch", color=t["muted"], fontsize=9)
    finish(fig, ax, t, "The citizen model matches cameras about 7° further north",
           "Average gap between the model's chance and the camera's aurora rate (9 cameras x 4 Kp bins).",
           "shift_error", mode)


def fig_shift_cameras(s, t, mode):
    s = s[s["camera"] != "kho svalbard"].reset_index(drop=True)
    if mode == "light":  # numbers quoted under this figure in CAMERA_VALIDATION.md
        for col, name in [("m1_at", "under the camera"), (f"m1_south", f"{DELTA}° south")]:
            print(f"shift_by_camera, {name}: mean gap {(s[col] - s['rate']).abs().mean():.2f}, "
                  f"correlation with camera rate {s[col].corr(s['rate']):.2f}")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.7, bottom=0.17)
    style(ax, t)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(s))
    for xi, a, b in zip(x, s["m1_at"], s["m1_south"]):
        if abs(a - b) < 0.03:
            continue
        ax.annotate("", xy=(xi, b), xytext=(xi, a),
                    arrowprops=dict(arrowstyle="-|>", color=t["ref"], linewidth=1.2, shrinkA=6, shrinkB=6))
    ax.scatter(x, s["m1_at"], s=55, marker="^", color=t["aqua"], edgecolor=t["surface"], linewidth=1.5, zorder=3,
               label="Citizen model, person under the camera")
    ax.scatter(x, s["m1_south"], s=55, marker="s", color=t["orange"], edgecolor=t["surface"], linewidth=1.5,
               zorder=3, label=f"Citizen model, person {DELTA}° south")
    ax.scatter(x, s["rate"], s=70, marker="o", color=t["blue"], edgecolor=t["surface"], linewidth=1.5, zorder=4,
               label="Camera: aurora overhead")
    ax.set_xticks(x, [f"{NAMES[c]}\n{m:.0f}°" for c, m in zip(s["camera"], s["mlat"])], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8, frameon=False, labelcolor=t["text"],
              handletextpad=0.3, columnspacing=1.2)
    finish(fig, ax, t, f"Moving the person {DELTA}° south closes most of the gap at the southern cameras",
           "Average over each camera's minutes. At the Scandinavian cameras it doesn't help (where the two model points\ncoincide, the triangle is hidden behind the square).",
           "shift_by_camera", mode)


def fig_precision(s, t, mode):
    s = s[(s["n_sure"] >= 1000) & (s["camera"] != "kho svalbard")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.76, bottom=0.12)
    style(ax, t)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(s))[::-1]
    ax.barh(y + 0.17, s["rate"], height=0.3, color=t["ref"], label="Aurora overall")
    ax.barh(y - 0.17, s["precision"], height=0.3, color=t["blue"], label="When the citizen model says > 70%")
    for yi, p in zip(y, s["precision"]):
        ax.text(p + 0.01, yi - 0.17, f"{p:.0%}", va="center", fontsize=8.5, color=t["text"])
    ax.set_yticks(y, [f"{NAMES[c]} ({m:.0f}°)" for c, m in zip(s["camera"], s["mlat"])], color=t["muted"])
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of minutes the camera saw aurora overhead", color=t["muted"], fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False, labelcolor=t["text"])
    finish(fig, ax, t, "When the citizen model is confident, there is aurora",
           f"Model for a person {DELTA}° south of each camera. Cameras with fewer than 1,000 confident minutes left out.",
           "precision_70", mode)


def main():
    OUT.mkdir(exist_ok=True)
    s, curve = load()
    print(s.round(3).to_string(index=False))
    print(curve.round(3).to_string())
    for mode, t in THEMES.items():
        fig_auc(s, t, mode)
        fig_sketch(t, mode)
        fig_shift_error(curve, t, mode)
        fig_shift_cameras(s, t, mode)
        fig_precision(s, t, mode)


if __name__ == "__main__":
    main()
