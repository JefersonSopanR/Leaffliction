import sys
import os
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tiff", ".JPG", ".JPEG", ".PNG",
}

def plot_counter(counts: Counter, title: str) -> None:
    """
    Render a bar chart (left) and pie chart (right) side by side.
    title is used as the figure heading.
    """
    classes = counts.keys()
    values  = counts.values()
    total   = sum(values)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), facecolor="#1A1A2E")
    fig.suptitle(
        f"Image Distribution — {title}   (total: {total})",
        fontsize=15, fontweight="bold", color="white",
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.set_facecolor("#0F0F23")

    bars = ax_bar.bar(classes, values, width=0.75)

    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_width() + bar.get_x() - (bar.get_width() / 2),
            bar.get_height() + 13,
            str(val),
            ha="center", va="bottom",
            fontsize=10, color="white", fontweight="bold",
        )

    ax_bar.set_xlabel("Disease class", color="white", labelpad=8)
    ax_bar.set_ylabel("Number of images", color="white", labelpad=8)
    ax_bar.set_title("Images per class", color="white", pad=10)
    ax_bar.tick_params(axis="x", colors="white")
    ax_bar.tick_params(axis="y", colors="white")
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray", zorder=0)

    # ── Pie chart ─────────────────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1])
    ax_pie.set_facecolor("#0F0F23")

    wedges, _, _ = ax_pie.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.6,
        wedgeprops={"linewidth": 1.5, "edgecolor": "#1A1A2E"},
    )

    ax_pie.set_title("Proportion per class", color="white", pad=10)
    ax_pie.legend(
        wedges,
        [f"{c}  ({v})" for c, v in zip(classes, values)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=8,
        labelcolor="white", facecolor="#0F0F23", edgecolor="#333355",
    )

    plt.tight_layout()
    plt.savefig('counter_image.png')



def main():
	argparser = argparse.ArgumentParser(description="Distribution /images")
	argparser.add_argument("images_dir", metavar="DIR", help="Images Directory")
	argparser.add_argument("plant_filter", nargs="?", help="plant_filter")
	
	args = argparser.parse_args()
	try:
		counter = Counter()
		plant_filter = args.plant_filter if args.plant_filter else None
		dir = Path(args.images_dir)
		if not dir.is_dir():
			raise Exception(f"{dir} is not a directory!")


		for d in os.scandir(dir):
			if not d.is_dir():
				continue
			if plant_filter and not d.name.startswith(plant_filter):
				print(f"{d} does not pass the plant filter {plant_filter}")
				continue

			summer = sum([1 for a in Path(d).iterdir() if a.is_file() and a.suffix in IMAGE_EXTENSIONS])
			
			if summer > 0:
				counter[d.name] = summer
		
		plot_counter(counter, plant_filter)


	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)

if __name__ == "__main__":
	main()