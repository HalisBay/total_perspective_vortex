import matplotlib.pyplot as plt


def plotter(raws):
    raws = list(raws)
    if not raws:
        return None

    raw = raws[0]
    raw.plot()
    plt.show()

