"""
Task-Utility-Aware Episodic Memory — Interactive Demo
Run with:  streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ─────────────────────────── page config ───────────────────────────
st.set_page_config(
    page_title="Robotic Memory Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CSV paths (single source of truth) ─────
DATA_DIR = os.path.join(os.path.dirname(__file__), "experiments")

CSV_OVERALL   = os.path.join(DATA_DIR, "milestone3_coverage_pilot", "validation_overall_hit_rate.csv")
CSV_CATEGORY  = os.path.join(DATA_DIR, "phase2_milestone2", "per_category_hit_rate.csv")
CSV_RESULTS   = os.path.join(DATA_DIR, "milestone3_coverage_pilot", "results.txt")

def _provenance(csv_path: str, note: str = ""):
    """Render a small grey data-source label."""
    rel = os.path.relpath(csv_path, os.path.dirname(__file__))
    st.caption(f"📂 **Data source:** `{rel}`   {note}")

@st.cache_data
def load_hit_rates():
    df = pd.read_csv(CSV_OVERALL, index_col=0)
    df.index.name = "policy"
    df.columns = [int(c) for c in df.columns]
    return df

@st.cache_data
def load_per_category():
    return pd.read_csv(CSV_CATEGORY)

hit_df   = load_hit_rates()
cat_df   = load_per_category()
BUDGETS  = [4, 8, 16, 32, 64, 128, 256]

POLICY_LABELS = {
    "uniform":                       "Uniform (Baseline)",
    "random":                        "Random",
    "stratified_random":             "Stratified Random",
    "pixel_novelty":                 "Top-K Pixel Novelty",
    "nms_pixel_novelty":             "NMS Pixel Novelty ✨",
    "motion_interaction":            "Top-K Motion Interaction",
    "nms_motion_interaction":        "NMS Motion Interaction ✨",
    "stratified_motion_interaction": "Stratified Motion",
    "oracle_greedy":                 "Oracle (Upper Bound)",
}

POLICY_COLORS = {
    "uniform":                       "#888888",
    "random":                        "#aaaaaa",
    "stratified_random":             "#bbbbbb",
    "pixel_novelty":                 "#e07b39",
    "nms_pixel_novelty":             "#2196F3",
    "motion_interaction":            "#e07b39",
    "nms_motion_interaction":        "#4CAF50",
    "stratified_motion_interaction": "#66BB6A",
    "oracle_greedy":                 "#9C27B0",
}

# ─────────────────────────── sidebar ────────────────────────────────
st.sidebar.title("🤖 Memory Policy Demo")
page = st.sidebar.radio("Choose demo section:", [
    "🏠 Project Overview",
    "🎬 Memory Policy Simulator",
    "📉 Budget vs Hit Rate",
    "⏱️ Temporal Bunching Visualizer",
    "🌊 Online Streaming Eviction",
    "📊 Task-Type Breakdown",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Task-Utility-Aware Episodic Memory**  
Foundation of Robotics — Course Project  
*Aditya Kishore, Ranjeet Kulkarni et al.*
""")

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.title("🤖 Task-Utility-Aware Episodic Memory Compression")
    st.markdown("### *How should a robot decide what to remember?*")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries Evaluated", "5,870", help="FindingDory validation set, 100 episodes")
    col2.metric("NMS Hit Rate @ B=4",      "0.593", delta="+0.313 vs Uniform")
    col3.metric("Oracle Gap Recovered",    "71.3%", delta="at Budget = 4")
    _provenance(CSV_OVERALL, "— metrics above from Milestone 3 validation run")

    st.markdown("---")
    st.markdown("""
## The Problem
A robot running for 30+ minutes sees **thousands of frames**.  
It can only store **B frames** (the memory budget).  
Later a user asks: *"Where did I leave my keys?"*  
The robot must already have the right frame — but it **couldn't have known** which one when recording.

> **Core question:** How do we choose which frames to keep *before* knowing the future question?
""")

    st.info("💡 Use the sidebar to explore each section of the demo.", icon="👈")

    st.markdown("---")
    st.markdown("## Key Finding at a Glance: Temporal Bunching")

    # Illustrative bunching diagram (synthetic, for illustration only)
    rng = np.random.default_rng(42)
    n_frames = 300
    scores = np.zeros(n_frames)
    scores[40:80]   += rng.exponential(0.5, 40)
    scores[50:60]   += 2.0
    scores[200:230] += rng.exponential(0.3, 30)
    scores += rng.exponential(0.05, n_frames)

    topk_idx = np.argsort(scores)[-8:]
    nms_idx, suppressed = [], set()
    for i in np.argsort(scores)[::-1]:
        if i not in suppressed and len(nms_idx) < 8:
            nms_idx.append(i)
            suppressed.update(range(max(0, i-20), min(n_frames, i+20)))
    nms_idx = sorted(nms_idx)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.barh(["Top-K (bunched)"], [1], left=0, color="#f5f5f5", height=0.4)
    ax.barh(["NMS (spread)"],   [1], left=0, color="#f5f5f5", height=0.4)
    for i in topk_idx:
        ax.barh(["Top-K (bunched)"], [0.012], left=i/n_frames, color="#e07b39", height=0.35)
    for i in nms_idx:
        ax.barh(["NMS (spread)"],   [0.012], left=i/n_frames, color="#4CAF50",  height=0.35)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Episode timeline →")
    ax.set_title("Same 8 frames selected: Top-K clusters in one spot, NMS spreads them out")
    ax.spines[["top","right","left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    st.caption("⚠️ Diagram above is illustrative (synthetic data). Real measurement: Top-K covers 39% of timeline at B=32 vs 100% for uniform. Source: `experiments/milestone3_coverage_pilot/results.txt`")


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — Memory Policy Simulator
# ════════════════════════════════════════════════════════════════════
elif page == "🎬 Memory Policy Simulator":
    st.title("🎬 Memory Policy Simulator")
    st.markdown("""
Pick a policy and a budget. The top chart shows a **simulated episode** with the selected frames as coloured vertical lines.
The hit rate number shown is **real** — fetched directly from the CSV for that policy and budget.
""")
    _provenance(CSV_OVERALL, "— hit rate numbers come from this CSV")

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        policy = st.selectbox("Memory Policy", options=list(POLICY_LABELS.keys()),
                               format_func=lambda k: POLICY_LABELS[k])
        budget = st.select_slider("Memory Budget B (frames)", options=BUDGETS, value=8)
        n_frames_sim = st.slider("Episode length (frames, for illustration)", 50, 500, 200, step=50)
        seed = st.number_input("Random seed", value=42, step=1)

    rng = np.random.default_rng(int(seed))
    frames = np.arange(n_frames_sim)
    scores = np.zeros(n_frames_sim)
    for _ in range(rng.integers(2, 6)):
        c = rng.integers(0, n_frames_sim)
        w = rng.integers(5, 20)
        h = rng.exponential(1.0)
        scores += h * np.exp(-0.5 * ((frames - c) / w)**2)
    scores += rng.exponential(0.05, n_frames_sim)

    if policy == "uniform":
        indices = np.linspace(0, n_frames_sim-1, budget, dtype=int)
    elif policy == "random":
        indices = sorted(rng.choice(n_frames_sim, budget, replace=False))
    elif policy == "stratified_random":
        segs = np.array_split(frames, budget)
        indices = sorted([rng.choice(s) for s in segs])
    elif policy in ("pixel_novelty", "motion_interaction"):
        indices = sorted(np.argsort(scores)[-budget:])
    elif policy in ("nms_pixel_novelty", "nms_motion_interaction"):
        window = max(1, n_frames_sim // (budget * 2))
        selected, suppressed = [], set()
        for i in np.argsort(scores)[::-1]:
            if i not in suppressed and len(selected) < budget:
                selected.append(i)
                suppressed.update(range(max(0, i-window), min(n_frames_sim, i+window)))
        indices = sorted(selected)
    elif policy == "stratified_motion_interaction":
        segs = np.array_split(frames, budget)
        indices = sorted([s[np.argmax(scores[s])] for s in segs])
    else:
        indices = sorted(np.argsort(scores)[-budget:])

    seg_hits = len(set(int(i * budget / n_frames_sim) for i in indices))
    coverage = seg_hits / budget

    with col_result:
        real_hr = hit_df.loc[policy, budget] if policy in hit_df.index else None
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames Kept", budget)
        m2.metric("Timeline Coverage", f"{coverage*100:.0f}%")
        if real_hr is not None:
            m3.metric("Real Hit Rate (from CSV)", f"{real_hr:.3f}",
                      delta=f"{real_hr - hit_df.loc['uniform', budget]:+.3f} vs Uniform")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [2, 1]})
        ax1.fill_between(frames, scores, alpha=0.3, color="#888")
        ax1.plot(frames, scores, color="#888", linewidth=0.8, alpha=0.5)
        for i in indices:
            ax1.axvline(i, color=POLICY_COLORS.get(policy, "#2196F3"), linewidth=2.0, alpha=0.85)
        ax1.set_ylabel("Saliency Score")
        ax1.set_xlim(0, n_frames_sim)
        ax1.set_title(f"{POLICY_LABELS[policy]} — {budget} frames selected  (simulated episode)")
        ax1.spines[["top","right"]].set_visible(False)
        kept_arr = np.zeros(n_frames_sim)
        for i in indices: kept_arr[i] = 1
        ax2.imshow(kept_arr.reshape(1,-1), aspect="auto", cmap="Greens", vmin=0, vmax=1,
                   extent=[0, n_frames_sim, 0, 1])
        ax2.set_yticks([])
        ax2.set_xlabel("Frame index →")
        ax2.set_title("Memory Buffer (green = kept)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("⚠️ Episode saliency curve is synthetic (for illustration). Hit rate number is real.")

    st.markdown("---")
    st.markdown(f"#### All policies at B = {budget} — real validation data")
    _provenance(CSV_OVERALL)

    comp = hit_df[budget].sort_values(ascending=False).reset_index()
    comp.columns = ["Policy", "Hit Rate"]
    comp["Policy Label"] = comp["Policy"].map(POLICY_LABELS).fillna(comp["Policy"])

    fig2, ax = plt.subplots(figsize=(9, 3.5))
    colors = []
    for p in comp["Policy"]:
        if p == policy:
            colors.append(POLICY_COLORS.get(p, "#2196F3"))
        elif "nms" in p:
            colors.append("#4CAF50")
        elif "oracle" in p:
            colors.append("#9C27B0")
        else:
            colors.append("#cccccc")
    bars = ax.barh(comp["Policy Label"].tolist(), comp["Hit Rate"].tolist(),
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Hit Rate (↑ better)")
    unif_val = comp.loc[comp["Policy"] == "uniform", "Hit Rate"].values
    if len(unif_val):
        ax.axvline(unif_val[0], color="#888", linestyle="--", linewidth=1.0, label="Uniform baseline")
    ax.set_title(f"Hit Rate at Budget B={budget}  (100 validation episodes, 5,870 queries)")
    ax.spines[["top","right"]].set_visible(False)
    for bar, v in zip(bars, comp["Hit Rate"].tolist()):
        ax.text(float(v) + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — Budget vs Hit Rate
# ════════════════════════════════════════════════════════════════════
elif page == "📉 Budget vs Hit Rate":
    st.title("📉 Budget vs Hit Rate — 100% Real Results")
    st.markdown("Every point on this chart is a real experiment — no simulation, no interpolation.")
    _provenance(CSV_OVERALL, "— 100 FindingDory validation episodes, 5,870 queries")

    show_policies = st.multiselect(
        "Select policies to display:",
        options=list(POLICY_LABELS.keys()),
        default=["uniform", "nms_motion_interaction", "motion_interaction", "oracle_greedy"],
        format_func=lambda k: POLICY_LABELS[k],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for p in show_policies:
        if p in hit_df.index:
            vals = hit_df.loc[p, BUDGETS].values
            lw = 2.5 if ("nms" in p or "oracle" in p) else 1.5
            ls = "--" if p == "oracle_greedy" else "-"
            ax.plot(BUDGETS, vals, marker="o", linewidth=lw, linestyle=ls,
                    color=POLICY_COLORS.get(p, "#aaa"), label=POLICY_LABELS[p], markersize=5)

    if "nms_motion_interaction" in show_policies and "oracle_greedy" in show_policies:
        nms_v = hit_df.loc["nms_motion_interaction", BUDGETS].values
        ora_v = hit_df.loc["oracle_greedy",          BUDGETS].values
        ax.fill_between(BUDGETS, nms_v, ora_v, alpha=0.1, color="#9C27B0", label="Remaining oracle gap")

    ax.set_xlabel("Memory Budget B (frames retained)", fontsize=12)
    ax.set_ylabel("Selection Hit Rate", fontsize=12)
    ax.set_title("Memory Policy Performance vs Budget\n(FindingDory Validation, 100 episodes, 5,870 queries)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Raw Numbers (from CSV)")
    display_df = hit_df.loc[[p for p in show_policies if p in hit_df.index], BUDGETS].copy()
    display_df.index = [POLICY_LABELS.get(p, p) for p in display_df.index]
    display_df.columns = [f"B={b}" for b in BUDGETS]
    st.dataframe(display_df.style.format("{:.3f}").highlight_max(axis=0, color="#c8e6c9"),
                 use_container_width=True)

    st.info("**Key numbers:** NMS Motion at B=4 → **0.593** vs Uniform **0.280** (+0.313, 95% CI [+0.274, +0.354]). Oracle gap recovered = **71.3%**.")


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — Temporal Bunching Visualizer
# ════════════════════════════════════════════════════════════════════
elif page == "⏱️ Temporal Bunching Visualizer":
    st.title("⏱️ Temporal Bunching Visualizer")
    st.markdown("""
**What is temporal bunching?**  
Top-K picks the highest-scoring frames — but those frames are often *near each other in time* (same event, same room, same second).
So you spend your whole budget on one moment and miss everything else.

**What NMS does:**  
After picking a frame, it blocks a *time window* around it, forcing the next pick to come from somewhere else.
This spreads the budget across the whole episode.

> **Real measurement from your experiments:** At B=32, Top-K covers only **39%** of the timeline. NMS covers **100%**.
""")
    _provenance(CSV_RESULTS, "— coverage percentages come from results.txt (segment coverage table)")

    st.markdown("---")
    n_frames = st.slider("Episode length (frames, for illustration)", 100, 500, 300)
    budget   = st.select_slider("Budget B", options=BUDGETS, value=32)
    nms_win  = st.slider("NMS suppression window (frames)", 5, 60, 20)

    rng = np.random.default_rng(7)
    scores = np.zeros(n_frames)
    for center, width, h in [(60,15,3.0),(65,10,2.5),(70,8,2.0),(200,20,1.0),(n_frames-50,12,1.5)]:
        x = np.arange(n_frames)
        scores += h * np.exp(-0.5*((x-center)/width)**2)
    scores += rng.exponential(0.1, n_frames)

    topk_idx = sorted(np.argsort(scores)[-budget:])
    nms_idx, suppressed = [], set()
    for i in np.argsort(scores)[::-1]:
        if i not in suppressed and len(nms_idx) < budget:
            nms_idx.append(i)
            suppressed.update(range(max(0, i-nms_win), min(n_frames, i+nms_win)))
    nms_idx = sorted(nms_idx)

    def seg_cov(idx, n, b):
        return len(set(int(i*b/n) for i in idx)) / b

    topk_cov = seg_cov(topk_idx, n_frames, budget)
    nms_cov  = seg_cov(nms_idx,  n_frames, budget)

    col1, col2 = st.columns(2)
    col1.metric("Top-K Timeline Coverage",  f"{topk_cov*100:.0f}%",
                delta=f"{(topk_cov-1.0)*100:+.0f}% vs full coverage", delta_color="inverse")
    col2.metric("NMS Timeline Coverage",    f"{nms_cov*100:.0f}%",
                delta=f"{(nms_cov-topk_cov)*100:+.0f}% vs Top-K")

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2,1,1]})
    x = np.arange(n_frames)
    axes[0].fill_between(x, scores, alpha=0.25, color="#888")
    axes[0].plot(x, scores, color="#666", linewidth=0.8)
    axes[0].set_ylabel("Saliency Score")
    axes[0].set_title("Episode Saliency Scores  (illustrative — same algorithm as real experiments)")
    axes[0].spines[["top","right"]].set_visible(False)

    tk_arr = np.zeros(n_frames)
    for i in topk_idx: tk_arr[i] = 1
    axes[1].imshow(tk_arr.reshape(1,-1), aspect="auto", cmap="Oranges", vmin=0, vmax=1,
                   extent=[0, n_frames, 0, 1])
    axes[1].set_yticks([])
    axes[1].set_ylabel("Top-K", fontsize=9, color="#e07b39")
    axes[1].set_title(f"Top-K: {budget} frames, {topk_cov*100:.0f}% coverage — clustered ⚠️",
                      color="#c0392b", fontsize=10)

    nms_arr = np.zeros(n_frames)
    for i in nms_idx: nms_arr[i] = 1
    axes[2].imshow(nms_arr.reshape(1,-1), aspect="auto", cmap="Greens", vmin=0, vmax=1,
                   extent=[0, n_frames, 0, 1])
    axes[2].set_yticks([])
    axes[2].set_ylabel("NMS", fontsize=9, color="#27ae60")
    axes[2].set_title(f"NMS: {budget} frames, {nms_cov*100:.0f}% coverage — spread out ✅",
                      color="#27ae60", fontsize=10)
    axes[2].set_xlabel("Frame index →")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.warning(
        f"**Illustrated result at B={budget}:** Top-K = {topk_cov*100:.0f}% coverage, "
        f"NMS = {nms_cov*100:.0f}% coverage.  "
        f"**Real experiment (from results.txt):** motion_interaction covers 41.8% of segments at B=32 vs NMS 100%."
    )


# ════════════════════════════════════════════════════════════════════
# PAGE 5 — Online Streaming Eviction
# ════════════════════════════════════════════════════════════════════
elif page == "🌊 Online Streaming Eviction":
    st.title("🌊 Online Streaming Eviction")

    st.markdown("""
## What is this? (plain English)

All the other policies (Top-K, NMS, Uniform) are **offline** — they see the whole episode first, then pick frames.
A real robot **cannot do that**. It sees one frame at a time and must decide *right now*.

**Online Streaming Eviction** solves this:

```
The robot has a box that holds B frames.

For every new frame that arrives:
  1. Score it (how interesting is it?)
  2. If the box is not full → put it in
  3. If the box IS full →
        Is new frame better than the WORST frame in the box?
           YES → throw out the worst, put new one in
           NO  → discard new frame, move on
```

That's the entire algorithm. The red dashed line in the animation below is the current frame.
Green vertical lines are what's currently in the buffer. Watch how the buffer changes as better frames arrive.

---
""")

    # Real numbers table — hardcoded from Colab run (no local CSV for streaming results)
    st.markdown("### Real Results — 100 Episodes, 5,870 Queries")
    st.markdown("""
| Budget B | Uniform (Baseline) | **Online Streaming Buffer** | Offline NMS (best offline) |
|:---:|:---:|:---:|:---:|
| 4  | 0.280 | **0.424** (+52% vs uniform) | 0.593 |
| 8  | 0.494 | **0.643** (+30% vs uniform) | 0.826 |
| 16 | 0.743 | **0.818** (+10% vs uniform) | 0.953 |
| 32 | 0.922 | **0.930** (+1%  vs uniform) | 0.983 |
""")
    st.caption(
        "⚠️ These numbers were computed live in Google Colab using `OnlineStreamingReservoir` on "
        "all 100 validation episodes. No local CSV exists for streaming results — "
        "they were computed from the FindingDory metadata (same source as the offline runs). "
        "The offline NMS column comes from `experiments/milestone3_coverage_pilot/validation_overall_hit_rate.csv`."
    )

    st.markdown("---")
    st.markdown("### Live Animation — Watch the Buffer Update Frame by Frame")
    st.markdown("The simulation below uses the **same eviction algorithm** as the real experiments.")

    budget_s = st.select_slider("Buffer size B", options=[4, 8, 16, 32], value=8)
    speed    = st.slider("Frames between screen updates (lower = slower/smoother)", 1, 20, 4)

    if st.button("▶ Run Streaming Simulation"):
        rng = np.random.default_rng(99)
        n_total = 150
        scores = np.zeros(n_total)
        for c, w, h in [(30,10,2.5),(80,15,1.8),(110,8,3.0),(140,12,1.5)]:
            x = np.arange(n_total)
            scores += h * np.exp(-0.5*((x-c)/w)**2)
        scores += rng.exponential(0.1, n_total)

        buffer_frames, buffer_scores = [], []
        eviction_log = []

        placeholder = st.empty()
        progress    = st.progress(0)
        log_txt     = st.empty()

        for t in range(n_total):
            s = float(scores[t])
            evicted = None
            if len(buffer_frames) < budget_s:
                buffer_frames.append(t)
                buffer_scores.append(s)
            else:
                min_i = int(np.argmin(buffer_scores))
                if s > buffer_scores[min_i]:
                    evicted = buffer_frames[min_i]
                    buffer_frames[min_i] = t
                    buffer_scores[min_i] = s
                    eviction_log.append(f"t={t}: evicted frame {evicted} (score {buffer_scores[min_i]:.2f} → {s:.2f})")

            if t % speed == 0 or t == n_total - 1:
                fig, axes = plt.subplots(2, 1, figsize=(10, 4),
                                         gridspec_kw={"height_ratios": [2, 1]})

                axes[0].fill_between(range(t+1), scores[:t+1], alpha=0.2, color="#888")
                axes[0].plot(range(t+1), scores[:t+1], color="#888", linewidth=0.8)
                for bf in buffer_frames:
                    axes[0].axvline(bf, color="#4CAF50", linewidth=2, alpha=0.85)
                axes[0].axvline(t, color="red", linewidth=1.5, linestyle="--",
                                label=f"Current frame t={t}")
                axes[0].set_xlim(0, n_total)
                axes[0].set_ylim(0, scores.max() * 1.1)
                axes[0].set_ylabel("Saliency Score")
                axes[0].set_title(
                    f"Streaming Buffer  |  t = {t}/{n_total-1}  |  "
                    f"Buffer: {len(buffer_frames)}/{budget_s} frames  |  "
                    f"Evictions so far: {len(eviction_log)}"
                )
                axes[0].legend(fontsize=8, loc="upper right")
                axes[0].spines[["top","right"]].set_visible(False)

                buf_arr = np.zeros(n_total)
                for bf in buffer_frames: buf_arr[bf] = 1
                axes[1].imshow(buf_arr.reshape(1,-1), aspect="auto", cmap="Greens",
                               vmin=0, vmax=1, extent=[0, n_total, 0, 1])
                # Mark current position
                axes[1].axvline(t, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
                axes[1].set_yticks([])
                axes[1].set_xlabel("Frame index →")
                axes[1].set_title(
                    f"Memory Buffer  (green = currently held,  red = current frame,  "
                    f"held frames: {sorted(buffer_frames)})",
                    fontsize=8
                )

                plt.tight_layout()
                placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
                progress.progress((t+1)/n_total)
                if eviction_log:
                    log_txt.text("Last eviction: " + eviction_log[-1])
                time.sleep(0.04)

        st.success(f"✅ Done.  Final buffer (B={budget_s}): frames {sorted(buffer_frames)}")
        st.markdown("**Eviction log:**")
        if eviction_log:
            for line in eviction_log[-10:]:
                st.text(line)
        else:
            st.text("No evictions — budget was never full (episode too short relative to B).")
    else:
        st.info("👆 Press ▶ Run to watch the buffer update in real time.")


# ════════════════════════════════════════════════════════════════════
# PAGE 6 — Task-Type Breakdown
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Task-Type Breakdown":
    st.title("📊 Task-Type Breakdown — Cross-Task Analysis")
    st.markdown("""
FindingDory has **3 high-level task categories** and **11 sub-categories**.
Different tasks stress-test different aspects of memory:
- **Spatial tasks** (Interaction, Object Attributes, Room Visitation…) — need frames where something happened
- **Temporal tasks** (Interaction Order, Duration Tracking…) — need frames spread across time
- **Multi-goal tasks** (Ordered/Unordered Revisitation) — need coverage of multiple events
""")
    _provenance(CSV_CATEGORY, "— all bars and numbers come from this CSV (Milestone 2 run, 100 validation episodes)")

    budget_c = st.select_slider("Budget", options=[4, 8, 16, 32, 64], value=8, key="cat_budget")
    policy_c = st.selectbox(
        "Policy", options=list(POLICY_LABELS.keys()),
        format_func=lambda k: POLICY_LABELS[k],
        index=list(POLICY_LABELS.keys()).index("nms_motion_interaction"),
        key="cat_policy"
    )

    subset = cat_df[(cat_df["budget"] == budget_c) & (cat_df["policy"] == policy_c)].copy()
    unif   = cat_df[(cat_df["budget"] == budget_c) & (cat_df["policy"] == "uniform")].copy()

    if subset.empty:
        st.warning(f"No data in CSV for policy=`{policy_c}` at budget={budget_c}. "
                   f"Try budget 4 or 8, or switch to nms_motion_interaction.")
    else:
        merged = subset.merge(unif[["low_level_category","hit_rate"]],
                              on="low_level_category", suffixes=("","_uniform"))
        merged["delta"] = merged["hit_rate"] - merged["hit_rate_uniform"]
        merged = merged.sort_values("hit_rate", ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#4CAF50" if d >= 0 else "#e74c3c" for d in merged["delta"]]
        bars = ax.barh(merged["low_level_category"].tolist(),
                       merged["hit_rate"].tolist(),
                       color=colors, alpha=0.85, edgecolor="white",
                       label=POLICY_LABELS[policy_c])
        ax.barh(merged["low_level_category"].tolist(),
                merged["hit_rate_uniform"].tolist(),
                color="none", edgecolor="#888", linewidth=1.5, linestyle="--",
                label="Uniform baseline")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Hit Rate")
        ax.set_title(f"{POLICY_LABELS[policy_c]}  vs  Uniform  |  B={budget_c}  "
                     f"(green = better than uniform, red = worse)")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        for bar, v in zip(bars, merged["hit_rate"].tolist()):
            ax.text(float(v)+0.005, bar.get_y()+bar.get_height()/2,
                    f"{v:.2f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        display = (merged[["high_level_category","low_level_category",
                            "hit_rate_uniform","hit_rate","delta"]]
                   .rename(columns={"high_level_category": "Category",
                                    "low_level_category":  "Sub-category",
                                    "hit_rate_uniform":    "Uniform HR",
                                    "hit_rate":            f"{POLICY_LABELS[policy_c]} HR",
                                    "delta":               "Δ (policy − uniform)"})
                   .set_index("Sub-category"))
        st.dataframe(
            display.style
            .format({"Uniform HR": "{:.3f}",
                     f"{POLICY_LABELS[policy_c]} HR": "{:.3f}",
                     "Δ (policy − uniform)": "{:+.3f}"})
            .background_gradient(subset=["Δ (policy − uniform)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
            use_container_width=True
        )
