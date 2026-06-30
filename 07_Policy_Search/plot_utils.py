import matplotlib.pyplot as plt
import seaborn as sns


def plot_trainingsinformation(data,
                              data_names,
                              colors,
                              figsize=(15, 4),
                              ylim=3000,
                              ylim_low=0,
                              columns=['Rewards', 'Timesteps per episode', 'Average score over 100 episodes'],
                              smoothing_factor=0.05,
                              alpha_non_smooth=0.3):
    """
    Plots key training information for one or more agents over time.

    Parameters:
    ----------
    data : list of pd.DataFrame
        A list containing pandas DataFrames, each representing training metrics for a different agent.
    data_names : list of str
        List of labels corresponding to each DataFrame in `data`.
    colors : list of str
        List of colors used for plotting each agent's data.
    figsize : tuple, optional (default=(15, 4))
        Size of the entire plot figure.
    ylim : int or float, optional (default=3000)
        Upper limit for the y-axis in the rewards plot.
    ylim_low : int or float, optional (default=0)
        Lower limit for the y-axis in the rewards plot. Use 0 for environments
        with non-negative rewards (e.g. CartPole) and a negative value for
        environments with negative rewards (e.g. Pendulum).
    columns : list of str, optional
        Names of the columns to be plotted from the DataFrames.
    smoothing_factor : float, optional (default=0.05)
        Smoothing factor for the exponential weighted moving average.
        Applied to all columns except the last one.
    alpha_non_smooth : float, optional (default=0.3)
        Transparency level for the unsmoothed lines.

    Returns:
    -------
    None
        Displays the plot with training metrics.
    """
    fig, ax = plt.subplots(figsize=figsize, ncols=len(columns), nrows=1, squeeze=False)
    # squeeze=False keeps ax a 2D array even for a single column; flatten to 1D
    # so a single-column plot (e.g. comparing only the average score) still works.
    ax = ax.ravel()

    for i, col in enumerate(columns):
        # The last column is plotted at full opacity without extra smoothing
        # (its raw curve is already the smoothed 100-episode average).
        is_last = (i == len(columns) - 1)
        alpha = 1 if is_last else alpha_non_smooth

        for k, df in enumerate(data):
            # Give the raw line a label only on the last column (or when it is
            # the only column); otherwise the smoothed line below carries it.
            label = data_names[k] if is_last else None
            sns.lineplot(df, x=df.index, y=col, alpha=alpha, color=colors[k],
                         label=label, ax=ax[i])

        if not is_last:
            for k, df in enumerate(data):
                sns.lineplot(df.ewm(alpha=smoothing_factor).mean(), x=df.index, y=col,
                             label=data_names[k], color=colors[k], ax=ax[i])

        ax[i].grid(alpha=0.3)
        ax[i].set_xlabel('Episodes')

        # Keep the legend only on the last axis; remove it from the others.
        legend = ax[i].get_legend()
        if not is_last and legend is not None:
            legend.remove()

    ax[-1].legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncols=len(data))

    ax[0].set_ylim(ylim_low, ylim)
