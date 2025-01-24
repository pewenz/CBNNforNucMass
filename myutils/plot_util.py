import matplotlib.pyplot as plt
from config import MAGIC_NUMBERS
from matplotlib.ticker import AutoMinorLocator


# Set the style of the chart
def set_chart_style(ax: plt.Axes):

    for y in MAGIC_NUMBERS:
        ax.axhline(y=y, color='black', linestyle='-.', linewidth=0.5)
    for x in MAGIC_NUMBERS:
        ax.axvline(x=x, color='black', linestyle='-.', linewidth=0.5)

    ax.set_xlim(0, 170)
    ax.set_xticks(range(0, 171, 20), minor=False)
    ax.set_xticks(range(10, 171, 20), minor=True)
    ax.set_xticklabels(range(0, 171, 20), fontsize=20)

    ax.set_ylim(0, 120)
    ax.set_yticks([20, 40, 60, 80, 100, 120], minor=False)
    ax.set_yticks([10, 30, 50, 70, 90, 110], minor=True)
    ax.set_yticklabels([20, 40, 60, 80, 100, 120], fontsize=20)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_edgecolor('black')

    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True, labelleft=True,
                   labelright=False, labeltop=False, direction='in')
    ax.tick_params(which='major', length=12, width=2)
    ax.tick_params(which='minor', length=7, width=2)


def plot_sample_nuclides_distribution(N_train, Z_train, N_test, Z_test):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    ax.scatter(N_train, Z_train, c='gray', marker='s', s=10, label='Training set')
    ax.scatter(N_test, Z_test, c='red', marker='o', s=10, label='Test set')

    set_chart_style(ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0.035, 0.965), prop={'size': 30}, scatterpoints=1, handletextpad=0,
              labelspacing=1, markerscale=5)
    ax.set_xlabel('N', fontsize=30)
    ax.set_ylabel('Z', fontsize=30)
    ax.set_title('Sample Distribution in the Chart of Nuclides', fontsize=30)

    ax.grid(False)

    return fig


# Plot the learning curve of the training process
# Parameters except `train_proportion` of this function must all be `list`
# e.g., train_proportion = 0.8
# `optimal = True` means to find the optimal fitting stage which has the minimum test RMSD
# `optimal = False` means to find the minimum train/test/entire RMSD through the whole training process


def plot_learning(train_proportion, find_optimal, epochs, train_rmsd, test_rmsd, train_rmsrd, test_rmsrd, lr_changes_at):

    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    entire_rmsd = []
    for i in range(len(train_rmsd)):
        entire_rmsd.append((train_proportion * train_rmsd[i] ** 2 + (1 - train_proportion) * test_rmsd[i] ** 2) ** 0.5)

    entire_rmsrd = []
    for i in range(len(train_rmsrd)):
        entire_rmsrd.append((train_proportion * train_rmsrd[i] ** 2 + (1 - train_proportion) * test_rmsrd[i] ** 2) ** 0.5)

    # First subplot
    axs[0].plot(epochs, train_rmsd, label='Train', marker='o', markersize=3)
    axs[0].plot(epochs, test_rmsd, label='Test', marker='s', c='r', markersize=3)
    axs[0].plot(epochs, entire_rmsd, label='Entire', marker='P', c='orange', markersize=3)

    # Second subplot
    axs[1].plot(epochs, train_rmsrd, label='Train', marker='o', markersize=3)
    axs[1].plot(epochs, test_rmsrd, label='Test', marker='s', c='r', markersize=3)
    axs[1].plot(epochs, entire_rmsrd, label='Entire', marker='P', c='orange', markersize=3)

    # Common settings for both subplots
    for ax in axs:
        for lr_change in lr_changes_at:
            ax.axvline(x=lr_change, color='black', linestyle='--', linewidth=0.5)

        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True, labelleft=True,
                       labelright=False, labeltop=False, direction='in')
        ax.set_xlabel('Epochs', fontsize=20)

        ax.legend(fontsize=25)
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), prop={'size': 25})

        ax.set_xlim(left=0 if epochs[0] < 5000 else epochs[0]-5000, right=epochs[-1]+5000)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='major', length=12, width=2)
        ax.tick_params(which='minor', length=7, width=2)

    # Set line width of the spines for both subplots
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    test_line = round(min(test_rmsd), 3)
    optimal_epoch = epochs[test_rmsd.index(min(test_rmsd))]
    if find_optimal:
        train_line = round(train_rmsd[test_rmsd.index(min(test_rmsd))], 3)
        entire_line = round(entire_rmsd[test_rmsd.index(min(test_rmsd))], 3)
    else:
        train_line = round(min(train_rmsd), 3)
        entire_line = round(min(entire_rmsd), 3)

    # Specific settings for each subplot
    axs[0].set_ylabel('RMSD (MeV)', size=20)
    axs[0].axhline(y=train_line, color='#4277EB', linestyle='-', linewidth=1)
    axs[0].annotate(f'{train_line}', xy=(axs[0].get_xlim()[1], train_line), xytext=(4, -6),
                    textcoords='offset points', fontsize=15,
                    color='#4277EB')
    axs[0].axhline(y=test_line, color='#FF1B1F', linestyle='-', linewidth=1)
    axs[0].annotate(f'{test_line}', xy=(axs[0].get_xlim()[1], test_line), xytext=(4, -6),
                    textcoords='offset points', fontsize=15,
                    color='#FF1B1F')
    axs[0].axhline(y=entire_line, color='#EBBF28', linestyle='-', linewidth=1)
    axs[0].annotate(f'{entire_line}', xy=(axs[0].get_xlim()[1], entire_line), xytext=(4, -4),
                    textcoords='offset points', fontsize=15,
                    color='#EBBF28')
    axs[0].axvline(x=optimal_epoch, color='green', linestyle='-', linewidth=3)

    axs[1].set_ylabel('RMSRD', fontsize=20)

    plt.tight_layout()

    plt.show()
