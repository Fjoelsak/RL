import matplotlib.pyplot as plt
import seaborn as sns


def plot_trainingsinformation(data,
                              data_names,
                              colors,
                              figsize=(15, 4),
                              ylim=3000,
                              columns=['Rewards', 'Average score over 100 episodes', 'Epsilon over episodes'],
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
    columns : list of str, optional
        Names of the columns to be plotted from the DataFrames.
        Should include ['Rewards', 'Average score over 100 episodes', 'Epsilon over episodes'].
    smoothing_factor : float, optional (default=0.05)
        Smoothing factor for the exponential weighted moving average.
        Applied to all columns except for 'Epsilon over episodes'.
    alpha_non_smooth : float, optional (default=0.3)
        Transparency level for the unsmoothed lines.

    Returns:
    -------
    None
        Displays the plot with training metrics.
    """
    fig, ax = plt.subplots(figsize=figsize, ncols=len(columns), nrows=1)

    for i, col in enumerate(columns):
        # Epsilon column is plotted at full opacity without smoothing
        alpha = alpha_non_smooth if i != 2 else 1

        for k, df in enumerate(data):
            sns.lineplot(df, x=df.index, y=col, alpha=alpha, color=colors[k], ax=ax[i])

        if i != 2:
            for k, df in enumerate(data):
                sns.lineplot(df.ewm(alpha=smoothing_factor).mean(), x=df.index, y=col,
                             label=data_names[k], color=colors[k], ax=ax[i])

        ax[i].grid(alpha=0.3)
        ax[i].set_xlabel('Episodes')

        if i == 0:
            ax[i].get_legend().remove()
        else:
            ax[1].legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncols=len(columns))

    ax[0].set_ylim(0, ylim)
