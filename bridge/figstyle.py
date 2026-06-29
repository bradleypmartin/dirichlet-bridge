r"""Shared Matplotlib font styling for the bridge figures.

The drivers render wide, multi-panel figures (often 15-17 in across) that are then
embedded in the preprint at a fraction of that width, which shrinks the on-page
text. ``enlarge()`` raises the default font sizes so titles, axis labels, tick
labels and legends stay legible once the figure is scaled down on the page.

Call it once, right after importing ``matplotlib.pyplot``, before any figure is
built. It only sets rcParams defaults, so any call that still passes an explicit
``fontsize=`` will override it -- the drivers drop those explicit sizes for the
text that should follow this style.
"""


def enlarge():
    """Bump the default Matplotlib font sizes for legible embedded figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        # Suptitles are long on these wide figures; keep them legible but not so
        # large that they overrun the figure width (21 pt clipped several of them).
        "figure.titlesize": 17,
    })
