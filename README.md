# Task-Utility-Aware Episodic Memory Compression for Long-Horizon Robotic Agents

A Foundation of Robotics research project on budget-constrained episodic memory for long-running egocentric robots under unknown future tasks.

---

## What Problem Are We Solving?

A robot moves around a house or warehouse for several hours, recording continuous egocentric video.
* It sees someone put keys on a table.
* It sees a pantry door opened.
* It sees a cup moved from the kitchen counter to the sink.
* It navigates through hundreds of unremarkable hallways and empty rooms.

Later, a user asks: **"Where did I leave my keys?"** or **"Did the robot put the cup in the sink?"**

The robot cannot afford to permanently store all 3 hours of video due to onboard flash limits, streaming bandwidth, and VLM context window costs. But **at the time of recording, the robot does not know what question it will be asked in the future.**

```
Continuous Egocentric Stream (Thousands of frames)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│    Task-Utility Memory Manager (Hard Budget B = 4 to 16)    │
│    - What frames should the robot keep?                     │
│    - What frames should it forget?                          │
│    - How to prevent storing duplicate/clustered views?      │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
Retained Episodic Memory (Small set of B high-utility frames)
                         │
                         ▼
Downstream VLM (e.g. Qwen2-VL) ──► Accurately Answers Future Questions
```

The mathematical objective is to maximize future task performance under a hard memory budget $B$:
$$\max_{M \subseteq O} \; \mathbb{E}_{\tau \sim p(\tau)} \Big[ U(\tau, M) \Big] \quad \text{subject to} \quad |M| \le B$$

---

## The Key Results: What Did We Find?

We evaluated all policies on the **FindingDory benchmark** (100 long-horizon validation episodes, **5,870 natural-language queries**, 60 task categories) under identical frame budgets $B \in [4, 256]$.

### 1. Numerical Hit Rate Table (Validation Split, 5,870 Queries)

| Policy | B = 4 | B = 8 | B = 16 | B = 32 | B = 64 | B = 128 | What It Means |
|---|---|---|---|---|---|---|---|
| **Recency (FIFO)** | 0.032 | 0.039 | 0.053 | 0.090 | 0.192 | 0.316 | **Catastrophic failure.** Keeping the last $B$ frames fails because queries ask about earlier events that FIFO discarded. |
| **Uniform Sampling** | 0.280 | 0.494 | 0.743 | 0.922 | 0.984 | 0.993 | Spreads frames evenly across time. Strong baseline, but blindly wastes memory on blank walls and empty floors. |
| **Top-K Novelty (Surprise)** | 0.396 | 0.563 | 0.692 | 0.817 | 0.881 | 0.942 | Frame-to-frame pixel change. Beats uniform at tiny budgets, but **drops below uniform from B=16 onward**. |
| **Top-K Motion Interaction** | 0.420 | 0.602 | 0.780 | 0.869 | 0.932 | 0.982 | Spatial concentration of change ($8 \times 8$ grid). Beats uniform at $B \le 16$, but **drops below uniform at B=32**. |
| **NMS Motion Interaction (Proposed)** | **0.593** | **0.826** | **0.953** | **0.983** | **0.988** | **0.991** | **Best practical method.** Enforces a temporal gap between picks. Beats uniform by **+31.3%** at $B=4$. |
| *Oracle Greedy (Upper Bound)* | *0.719* | *0.906* | *0.997* | *1.000* | *1.000* | *1.000* | Retrospective upper bound using ground-truth query labels (unachievable by real policies). |

---

## Why Did This Happen? The Core Discoveries

### 1. The "Temporal Bunching" Failure Mode
Why did Top-K novelty and Top-K interaction fall behind simple uniform sampling at $B \ge 32$?
* **The Reason:** If an interesting action (like picking up an object) lasts 10 seconds, the highest-scoring frames all come from that same 10-second window.
* **The Diagnostic:** Top-K picks cover only **$34\%–45\%$ of the episode timeline** (uniform covers $100\%$). It keeps 4 slightly different angles of the same grasp, while completely forgetting the remaining 25 minutes of the robot's run.

### 2. The Solution: Temporal NMS Closes 70%–80% of the Oracle Gap
Our policy sorts frames by interaction score, but suppresses any candidate within a temporal window $\tau_{\text{gap}} = \max\left(1, \frac{L}{2B}\right)$ of an already-accepted frame.
* At $B=4$: NMS reaches **$0.593$** vs uniform **$0.280$** (recovering **$71.3\%$** of the uniform-to-oracle distance).
* At $B=8$: NMS reaches **$0.826$** vs uniform **$0.494$** (recovering **$80.6\%$** of the distance).
* Tested with 2,000-resample bootstrap confidence intervals: the difference at $B=4$ is $+0.313$ $[+0.274, +0.354]$ ($p < 0.001$).

### 3. Downstream VLM Question-Answering (Qwen2-VL-2B-Instruct)
Selection hits are not just theoretical frame overlaps. Passing the retained frames to an open-weights Vision-Language Model on GPU showed that NMS memory directly boosts actual Question-Answering accuracy:
* **$B=4$ frames:** Uniform QA accuracy = **$21.3\%$** $\to$ NMS QA accuracy = **$48.6\%$** (+27.3% boost).
* **$B=8$ frames:** Uniform QA accuracy = **$39.5\%$** $\to$ NMS QA accuracy = **$69.4\%$** (**+29.9% boost**).

### 4. Real Video Case Study (`ep_1.mp4`)
Inspecting real video frames under a strict budget of $B=4$ frames:
* **Uniform Sampling:** Frame 0 captured a **completely blank beige wall** (wasting 25% of the memory budget); Frame 140 captured empty wooden flooring.
* **NMS Memory:** Frame 65 captured the dining area; Frame 298 captured the armchair clearly showing the **purple tape roll** resting on the seat; Frame 398 captured the robot arm manipulating it.
* When asked *"Where was the purple tape located before pickup?"*, Uniform memory failed completely, while NMS memory answered correctly.

---

## The Cross-Embodiment Perspective

A memory policy cannot assume a specific camera mounting or locomotion style:
1. **Wheeled Rover (FindingDory / Stretch):** Smooth planar motion; turns create broad uniform pixel changes.
2. **Manipulator Arm (Tabletop):** Static base; motion is localized to the gripper workspace.
3. **Walking Humanoid / Wearable:** High body bounce (1–2 Hz) and head saccades create continuous camera jitter.

**Why NMS Motion Interaction is Embodiment-Invariant:**
* **Spatial Variance Concentration:** Measures standard deviation across an $8 \times 8$ grid of block differences. Whole-camera motion (turning or head shaking) changes all blocks uniformly (low score). Hand-object interactions change only 4–6 blocks against a static background (high score).
* Evaluated across FindingDory's embodiment task domains at $B=4$:
  * *Mobile Base Navigation (Spatial):* Uniform = $0.312$ $\to$ NMS = **$0.582$** (+27.0%).
  * *Manipulator Arm Interaction (Multi-Goal):* Uniform = $0.245$ $\to$ NMS = **$0.635$** (+39.0%).

---

## Project Status

| Item | Status | Notes |
|---|---|---|
| **FindingDory Harness** | **Done** | Loader, Query parser, and selection-hit evaluation (`src/data/`, `src/eval/`) |
| **Baselines & Proxies** | **Done** | Uniform, Recency, Random, Top-K Novelty, Top-K Interaction, Oracle Greedy |
| **Milestone 1–3 Experiments** | **Done** | Tested on 100 validation episodes and 100 train sample episodes with 2,000 bootstrap CIs |
| **Downstream VLM Reader** | **Done** | Benchmarked with `Qwen2-VL-2B-Instruct` on GPU across budgets $B \in \{4, 8, 16\}$ |
| **Ablation Studies** | **Done** | Swept suppression window $\alpha \in [0.1, 2.0]$, confirming $\alpha^* = 0.5$ is optimal |
| **Cross-Embodiment Analysis** | **Done** | Evaluated across Navigation vs. Manipulation task domains and camera dynamics |
| **Final Technical Report** | **Done** | Comprehensive, self-contained report at [reports/Final_report.md](reports/Final_report.md) |
| **Streaming Reservoir Eviction** | Future Work | Currently runs offline per episode; online streaming buffer left as Phase 4 |

---

## Repository Structure

```
.
├── README.md                               This file
├── literature_review.xlsx                  Team literature review (P01–P16)
├── docs/
│   ├── PROJECT_CONTEXT.md                  Technical onboarding context
│   ├── project_proposal.md                 Original research proposal
│   ├── literature_analysis.md              8-paper deep comparative analysis
│   └── research_roadmap.md                 Gated seven-phase roadmap
├── src/
│   ├── data/findingdory.py                 Parquet metadata loader and Query data structures
│   ├── eval/selection_hit.py               Selection-hit scoring harness
│   ├── policies/baselines.py               Index-only policies (uniform, recency, random, oracle)
│   ├── policies/visual_signals.py          Signal policies (Top-K, Stratified, Temporal NMS)
│   └── scripts/
│       ├── download_findingdory.py         Downloads benchmark Parquet metadata (~1.4 MB)
│       ├── fetch_findingdory_videos.py     HTTP Range-fetch for selective video downloading
│       ├── extract_pixel_novelty.py        Computes frame-to-frame pixel difference
│       ├── extract_motion_concentration.py Computes 8x8 spatial motion concentration
│       ├── run_phase2_milestone1.py        Runs Milestone 1 index baselines
│       ├── run_phase2_milestone2.py        Runs Milestone 2 Top-K visual signals
│       └── run_milestone3_coverage.py      Runs Milestone 3 NMS coverage experiments
├── experiments/
│   ├── experiments_colab.ipynb             Master Google Colab notebook (VLM + Ablation + Plots)
│   ├── phase2_milestone1/                  Milestone 1 result CSVs
│   ├── phase2_milestone2/                  Milestone 2 result CSVs and bootstrap CIs
│   └── milestone3_coverage_pilot/          Milestone 3 result CSVs and results.txt
└── reports/
    ├── Final_report.md                     Complete, self-contained final capstone report
    └── figures/                            High-resolution publication figures
        ├── milestone3_overall_performance.png
        ├── cross_embodiment_generalization.png
        ├── downstream_vlm_accuracy.png
        ├── ablation_suppression_window.png
        └── visual_memory_comparison_grid.png
```

---

## How to Reproduce

### 1. Run Index Baselines (Local CPU, ~10 seconds)
Requires Python 3 with `pandas`, `pyarrow`, `numpy`, and `requests`:
```bash
python3 src/scripts/download_findingdory.py   # Downloads ~1.4 MB of metadata
python3 src/scripts/run_phase2_milestone1.py  # Runs generic baselines
```

### 2. Run GPU VLM Experiments & Visual Plots (Google Colab)
Open `experiments/experiments_colab.ipynb` in [Google Colab](https://colab.research.google.com/) with a free **T4 GPU runtime**. It executes:
1. Milestone 1 & 3 verification and plotting.
2. Cross-embodiment evaluation.
3. Live `Qwen2-VL-2B-Instruct` downstream Question-Answering benchmark.
4. Temporal suppression window ablation study.
5. Real video range-fetch and side-by-side visual comparison grid.

---

## Team

* **Atharva Narkhede**
* **Ranjeet Kulkarni**
* **Pratik Pansare**
* **Agraj Singh**  
* **Repository Owner / Lead:** Aditya Kishore  
*Course:* Foundation of Robotics, Semester Research Project.
