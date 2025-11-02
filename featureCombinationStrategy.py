import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")


# Function to draw rounded boxes
def draw_box(text, xy, color="#5DADE2"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        1.8,
        0.8,
        boxstyle="round,pad=0.2",
        linewidth=1.5,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.9,
        y + 0.4,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        weight="bold",
    )


# Boxes
draw_box("Left Foot\n56 Features", (0, 0))
draw_box("Right Foot\n56 Features", (3, 0))
draw_box("Concatenated\n112 Combined Features", (6, 0))
draw_box("PCA / Model\nInput", (9, 0))

# Arrows
for i in [1.8, 4.8, 7.8]:
    ax.arrow(
        i,
        0.4,
        0.8,
        0,
        width=0.02,
        head_width=0.3,
        head_length=0.3,
        fc="black",
        ec="black",
    )

# Title
plt.title(
    "Feature Combination Strategy – Left & Right Foot Concatenation",
    fontsize=13,
    weight="bold",
    pad=15,
)
plt.tight_layout()
plt.show()
