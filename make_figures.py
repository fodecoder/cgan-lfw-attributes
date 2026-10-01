"""Build the README figures from a finished training run (added in 2026).

Reuses the notebook's own code for the generator, so there's no second copy
of the model to drift. Needs output/generator.pth and the executed notebook
(for the training log), e.g.:

    .venv\\Scripts\\python make_figures.py run_2026_out.ipynb
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

NOTEBOOK = Path(__file__).with_name("cgan_lfw_attributes.ipynb")
DOCS = Path(__file__).with_name("docs")

# Imports, arguments and the Generator class, straight from the notebook.
cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
         if c["cell_type"] == "code"]
ns = {}
for marker in ("import torch.nn as nn", "LAT_DIM = ", "class Generator(nn.Module):"):
    exec(next(src for src in cells if marker in src), ns)

# Positions in LFWCrop.attributes_to_use (the LFWCrop dataset class).
SWEEP = {"Male": 0, "Blond Hair": 10, "Eyeglasses": 14, "Smiling": 17}
STEPS = np.linspace(-2, 2, 9)


def attribute_sweep(device):
    generator = ns["Generator"]().to(device)
    generator.load_state_dict(torch.load("output/generator.pth", map_location=device))
    generator.eval()

    torch.manual_seed(0)
    noise = torch.randn(1, ns["LAT_DIM"], device=device).repeat(len(STEPS), 1)
    rows = []
    with torch.no_grad():
        for index in SWEEP.values():
            labels = torch.zeros(len(STEPS), ns["N_CLASSES"], device=device)
            labels[:, index] = torch.tensor(STEPS, dtype=torch.float32, device=device)
            rows.append(generator(noise, labels, len(STEPS)).cpu())
    grid = vutils.make_grid(torch.cat(rows), nrow=len(STEPS), normalize=True, padding=2)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_xticks([(i + 0.5) * 34 for i in range(len(STEPS))])
    ax.set_xticklabels([f"{s:+.1f}" for s in STEPS])
    ax.set_yticks([(i + 0.5) * 34 for i in range(len(SWEEP))])
    ax.set_yticklabels(list(SWEEP))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("attribute value (all other attributes 0, same noise vector)")
    fig.tight_layout()
    fig.savefig(DOCS / "attribute_sweep.png", dpi=150)
    plt.close(fig)


def loss_curve(executed_notebook):
    log = ""
    for cell in json.loads(Path(executed_notebook).read_text(encoding="utf-8"))["cells"]:
        for out in cell.get("outputs", []):
            log += "".join(out.get("text", ""))
    rows = re.findall(r"\[Epoch: (\d+)/\d+\]\[D loss: ([\d.]+)\]\[G loss: ([\d.]+)\]", log)
    epoch, d_loss, g_loss = (np.array(col, dtype=float) for col in zip(*rows))
    (DOCS / "training_log_2026.txt").write_text(
        "\n".join(f"{int(e)}\t{d}\t{g}" for e, d, g in zip(epoch, d_loss, g_loss)) + "\n",
        encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
    for ax, values, name in ((axes[0], d_loss, "Discriminator loss"), (axes[1], g_loss, "Generator loss")):
        ax.plot(epoch, values, linewidth=1, color="#9ca3af")
        if len(values) >= 10:
            smooth = np.convolve(values, np.ones(10) / 10, mode="valid")
            ax.plot(epoch[9:], smooth, linewidth=2, color="#2563eb")
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(0.99, 0.01, "grey: last batch of each epoch · blue: 10-epoch mean",
             ha="right", va="bottom", fontsize=8, color="#6b7280")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(DOCS / "loss_curve.png", dpi=150)
    plt.close(fig)
    return int(epoch[-1])


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    attribute_sweep("cuda" if torch.cuda.is_available() else "cpu")
    print("epochs in log:", loss_curve(sys.argv[1]))
