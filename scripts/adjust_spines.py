# Matplotlib utility: offset axes spines for a cleaner look.
# http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

def adjust_spines(ax, spines, offset=10):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', offset))
        else:
            spine.set_color('none')

        if 'left' in spines:
            ax.yaxis.set_tick_params(length=5)
            ax.yaxis.set_tick_params(direction='out')
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_label_position('left')
        elif 'right' in spines:
            ax.yaxis.set_tick_params(length=5)
            ax.yaxis.set_tick_params(direction='out')
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
        else:
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_tick_params(length=5)
            ax.xaxis.set_tick_params(direction='out')
            ax.xaxis.set_label_position('bottom')
        elif 'top' in spines:
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_tick_params(length=5)
            ax.xaxis.set_tick_params(direction='out')
            ax.xaxis.set_label_position('top')
        else:
            ax.xaxis.set_ticks([])


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.random.random(100)
    fig = plt.figure(100)
    fig.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(x)
    adjust_spines(ax, ["left", "bottom"])
