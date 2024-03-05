import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import transformer_lens.utils as utils


def reduce_logit_mean(all_logits):
    """
    all_logits: [..., name, batch, answer]
    """
    return all_logits.mean(dim=-2)


def reduce_acc(all_logits, intended_answers):
    """
    all_logits: [..., name, batch, answer]
    intended_answers: [..., name]
    """

    intended_answers = torch.tensor(intended_answers)
    return (
        (all_logits.argmax(dim=-1) == intended_answers[..., None]).float().mean(dim=-1)
    )


def reduce_balanced_acc(all_logits, intended_answers, offsets=None):
    if offsets is None:
        # median is taken over name and batch
        median = torch.flatten(all_logits, start_dim=-3, end_dim=-2).quantile(
            dim=-2, q=0.5
        )  # we've lost both dimensions
        offsets = median[..., None, None, :]  # restore dimensions
    rebalanced_logits = all_logits - offsets
    return reduce_acc(rebalanced_logits, intended_answers)


def reduce_distractor_acc(tensor, intended_answers):
    """
    tensor: [..., name, batch, attr]
    """
    baseline = tensor[..., -1:, :, :].quantile(dim=-2, q=0.5, keepdim=True)
    mat = tensor[..., :-1, :, :]
    return reduce_balanced_acc(mat, intended_answers=intended_answers, offsets=baseline)


def heatmap(ax, tensor):
    sns.heatmap(
        utils.to_numpy(tensor),
        ax=ax,
        annot=True,
        fmt=".2f",
        square=True,
        vmin=-5,
        vmax=0,
        cbar=False,
        cmap=sns.color_palette("dark:#5A9", as_cmap=True),
    )


def plot_factorization(results):
    tens = reduce_logit_mean(results)
    num_row, num_col = 2, 2
    fig, axs = plt.subplots(num_row, num_col, figsize=(7, 7))
    for r in range(num_row):
        for c in range(num_col):
            ax = axs[r, c]
            heatmap(ax, tens[r, c, :, :])
            ax.set_title(f"$x_0={r}$, $x_1={c}$")
            ax.set_xticklabels(["$A_0$", "$A_1$", "$A_0'$", "$A_1'$"])
            ax.set_yticklabels(["$E_0$", "$E_1$", "$E_0'$", "$E_1'$"])
            if c == 0:
                ax.set_ylabel("Query name")
            if r == 1:
                ax.set_xlabel("Attributes")
