"""Render the agent architecture diagram to PNG (cover image) + SVG (writeup vector)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 1600, 1000
fig = plt.figure(figsize=(16, 10), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, fc, ec, r=14, lw=2):
    ax.add_patch(FancyBboxPatch((x, y + h), w, -h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1))

def txt(x, y, s, size=9, color="#0f172a", ha="left", weight="normal", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            fontweight=weight, fontstyle=style)

def arrow(x1, y1, x2, y2, color="#475569", lw=2, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=lw, color=color,
                 linestyle="--" if dashed else "-",
                 connectionstyle=f"arc3,rad={rad}"))

# Title
txt(60, 40, "Alzheimer's Biomarker Hypothesis Agent", size=25, weight="bold")
txt(60, 70, "An agentic research workflow — genetic risk \u00d7 imaging-biomarker progression  \u00b7  built on Google ADK + Gemini",
    size=12, color="#475569")

# Top band
box(60, 140, 190, 92, "#eef2ff", "#6366f1")
txt(155, 175, "Researcher", size=13, weight="bold", ha="center")
txt(155, 200, "free-text research", size=9, color="#334155", ha="center")
txt(155, 218, "question", size=9, color="#334155", ha="center")
arrow(250, 186, 318, 186)

box(322, 140, 210, 92, "#fff1f2", "#be123c")
txt(427, 172, "Safety guardrail", size=12, weight="bold", ha="center")
txt(427, 195, "check_query_safety", size=9, color="#334155", ha="center")
txt(427, 214, "refuses personal / clinical Qs", size=9, color="#334155", ha="center")
arrow(532, 186, 600, 186)

box(604, 128, 506, 116, "#4f46e5", "#3730a3")
txt(857, 160, "AI Agent \u00b7 Google ADK + Gemini", size=16, weight="bold", color="#f8fafc", ha="center")
txt(857, 189, "plans \u2192 selects one biomarker + one genotype scheme \u2192", size=10, color="#cbd5e1", ha="center")
txt(857, 209, "calls tools \u2192 returns a citation-backed hypothesis card", size=10, color="#cbd5e1", ha="center")
txt(1096, 233, "Day 1 \u00b7 agents", size=9.5, color="#cbd5e1", ha="right", style="italic")

# return arrow agent -> researcher
arrow(604, 148, 205, 138, color="#4f46e5", lw=2, dashed=True, rad=0.16)
txt(405, 108, "hypothesis card returned", size=9.5, color="#4f46e5", ha="center", style="italic")

# agent -> tools
arrow(760, 244, 260, 334); arrow(857, 244, 590, 334); arrow(950, 244, 930, 334)

# Tools band
box(100, 338, 300, 200, "#f0fdfa", "#0d9488")
txt(120, 362, "Ingestion tools", size=12, weight="bold")
txt(380, 362, "Day 2", size=9.5, color="#475569", ha="right", style="italic")
for i, s in enumerate(["load_dataset \u00b7 validate_schema", "clean_biomarkers (quarantine)",
                       "group_by_genotype", "annotate_variant"]):
    txt(120, 392 + i*22, s, size=9, color="#334155")
txt(120, 490, "\u2192 Ensembl / MyVariant / GWAS", size=9, color="#0f766e")
txt(120, 510, "   APIs, cached for offline use", size=9, color="#0f766e")

box(440, 338, 300, 200, "#f0fdfa", "#0d9488")
txt(460, 362, "Statistics engine", size=12, weight="bold")
txt(720, 362, "Day 3", size=9.5, color="#475569", ha="right", style="italic")
for i, s in enumerate(["compute_group_summary", "compute_subject_slopes",
                       "compare_biomarker_trajectories", "adjusted_effect (age + sex)",
                       "rank_biomarkers"]):
    txt(460, 392 + i*22, s, size=9, color="#334155")
txt(460, 514, "OLS + CIs, effect sizes (scipy)", size=9, color="#0f766e")

box(780, 338, 300, 200, "#f0fdf4", "#15803d")
txt(800, 362, "Hypothesis-card generator", size=12, weight="bold")
txt(800, 394, "Hypothesis \u00b7 Evidence", size=9.5, color="#334155", weight="bold")
txt(800, 416, "Caution \u00b7 Citations", size=9.5, color="#334155", weight="bold")
txt(800, 448, "curated citations only", size=9, color="#15803d")
txt(800, 468, "(no hallucinated references)", size=9, color="#15803d")
txt(800, 496, "auto cautions: synthetic,", size=9, color="#15803d")
txt(800, 516, "observational, not-for-dx", size=9, color="#15803d")

# tools -> data
arrow(250, 538, 250, 634); arrow(590, 538, 370, 634)
arrow(300, 500, 920, 634, color="#94a3b8", lw=1.6, dashed=True)

# Data band
box(100, 638, 300, 118, "#fffbeb", "#b45309")
txt(250, 674, "Dataset store", size=12, weight="bold", ha="center")
txt(250, 700, "in-memory + disk rehydration", size=9, color="#334155", ha="center")
txt(250, 720, "(survives kernel restarts)", size=9, color="#334155", ha="center")
txt(250, 742, "Day 3 \u00b7 memory", size=9.5, color="#475569", ha="center", style="italic")

box(440, 638, 300, 118, "#fffbeb", "#b45309")
txt(590, 674, "Synthetic ADNI-style data", size=12, weight="bold", ha="center")
txt(590, 700, "schema-validated, literature-", size=9, color="#334155", ha="center")
txt(590, 720, "calibrated effect directions", size=9, color="#334155", ha="center")
txt(590, 742, "ADNI / OASIS-3 ready", size=9, color="#b45309", ha="center")

box(780, 638, 300, 118, "#f8fafc", "#64748b")
txt(930, 674, "External bio-APIs", size=12, weight="bold", ha="center")
txt(930, 700, "Ensembl VEP \u00b7 MyVariant", size=9, color="#334155", ha="center")
txt(930, 720, "GWAS Catalog", size=9, color="#334155", ha="center")
txt(930, 742, "cached \u00b7 offline-safe", size=9, color="#64748b", ha="center")

# Right QA panel
box(1150, 128, 390, 628, "#0f172a", "#0f172a", r=16, lw=1)
txt(1345, 155, "Quality & Security", size=15, weight="bold", color="#f8fafc", ha="center")
txt(1345, 182, "Day 4", size=10, color="#cbd5e1", ha="center", style="italic")

box(1180, 212, 330, 196, "#1e293b", "#334155", r=12)
txt(1200, 238, "Eval harness \u00b7 17 / 17", size=12, weight="bold", color="#f8fafc")
for i, s in enumerate(["routing        6 / 6", "safety          6 / 6",
                       "grounding   3 / 3", "ranking        2 / 2"]):
    txt(1200, 268 + i*26, s, size=10, color="#cbd5e1")
txt(1200, 382, "100% pass rate", size=10, color="#86efac", weight="bold")

box(1180, 424, 330, 146, "#1e293b", "#334155", r=12)
txt(1200, 448, "Grounding checks", size=11, weight="bold", color="#f8fafc")
for i, s in enumerate(["\u2022 citations \u2286 curated set", "\u2022 reported stats = recomputed",
                       "\u2022 effect direction not inverted", "\u2022 mandatory cautions present"]):
    txt(1200, 474 + i*24, s, size=9.5, color="#cbd5e1")

box(1180, 586, 330, 146, "#3f1d2b", "#be123c", r=12)
txt(1200, 610, "Guardrail (defense-in-depth)", size=11, weight="bold", color="#f8fafc")
txt(1200, 636, "\u2022 system-instruction refusal", size=9.5, color="#cbd5e1")
txt(1200, 660, "\u2022 independent screen tool", size=9.5, color="#cbd5e1")
txt(1200, 690, "Research tool \u2014 NOT for", size=11, weight="bold", color="#fca5a5")
txt(1200, 712, "diagnosis or clinical use", size=11, weight="bold", color="#fca5a5")

# Footer legend
txt(60, 806, "Course concepts demonstrated:", size=12, weight="bold")
chips = [("Day 1 \u00b7 Agents", 60, 150, "#eef2ff", "#6366f1", "#0f172a"),
         ("Day 2 \u00b7 Tools & APIs", 222, 196, "#f0fdfa", "#0d9488", "#0f172a"),
         ("Day 3 \u00b7 Context & Memory", 430, 220, "#fffbeb", "#b45309", "#0f172a"),
         ("Day 4 \u00b7 Quality & Security", 662, 220, "#0f172a", "#0f172a", "#f8fafc"),
         ("Day 5 \u00b7 Packaging", 894, 176, "#f0fdf4", "#15803d", "#0f172a")]
for label, x, w, fc, ec, tc in chips:
    box(x, 828, w, 30, fc, ec, r=8, lw=1.5)
    txt(x + w/2, 844, label, size=9.5, color=tc, ha="center")

fig.savefig("/mnt/user-data/outputs/architecture_diagram.png", dpi=100, facecolor="white")
fig.savefig("/mnt/user-data/outputs/architecture_diagram.svg", facecolor="white")
print("wrote architecture_diagram.png and .svg")
