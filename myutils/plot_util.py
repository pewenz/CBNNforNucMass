import matplotlib.pyplot as plt
import config as cf
from config import MAGIC_NUMBERS, TRAIN_DATA_PATH
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import pandas as pd
import numpy as np
import torch


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


def normalize(data, min_val=None, max_val=None):
    """Normalize the data to [-1, 1]"""
    min_val = data.min() if min_val is None else min_val
    max_val = data.max() if max_val is None else max_val

    normalize_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalize_data, min_val, max_val


def full_pivot_table(dataframe, value):
    """Create a full pivot table with all Z and N values"""
    raw_data = pd.read_csv(TRAIN_DATA_PATH)
    max_proton_dim = raw_data['Z'].max()
    max_neutron_dim = raw_data['N'].max()

    full_index = pd.Index(np.arange(0, max_proton_dim + 1), name='Z')
    full_columns = pd.Index(np.arange(0, max_neutron_dim + 1), name='N')

    pivot_table = dataframe.pivot(index='Z', columns='N', values=value)
    pivot_table = pivot_table.reindex(index=full_index, columns=full_columns)

    return pivot_table


def plot_residual_heatmap(model, physical_model_name, scale=False):
    raw_data = pd.read_csv(TRAIN_DATA_PATH)
    heatmap_df = raw_data[['Z', 'N', physical_model_name]].copy()
    normalized_phy_mol_res, minimum, maximum = normalize(heatmap_df[physical_model_name])
    heatmap_df[f"normalized_{physical_model_name}"] = normalized_phy_mol_res

    pred_path = f"model/prediction/pred_{model.__class__.__name__}.csv"
    pred_df = pd.read_csv(pred_path)
    phy_NN_res = heatmap_df[physical_model_name] - pred_df['pred_NN']
    heatmap_df['phy_NN_res'] = phy_NN_res
    normalized_phy_NN_res = normalize(phy_NN_res, minimum, maximum)[0]
    heatmap_df["normalized_phy_NN_res"] = normalized_phy_NN_res

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 24))

    plt.subplots_adjust(hspace=0.25)

    if scale:
        phy_res_pivot = full_pivot_table(heatmap_df, f"normalized_{physical_model_name}")
        phy_NN_res_pivot = full_pivot_table(heatmap_df, "normalized_phy_NN_res")

        sns.heatmap(phy_res_pivot, cmap='seismic', center=0, cbar_kws={'pad': 0.02}, vmin=-1, vmax=1,
                    ax=ax1).set_facecolor('#CCCCCC')
        sns.heatmap(phy_NN_res_pivot, cmap='seismic', center=0, cbar_kws={'pad': 0.02}, vmin=-1, vmax=1,
                    ax=ax2).set_facecolor('#CCCCCC')
    else:
        phy_res_pivot = full_pivot_table(heatmap_df, physical_model_name)
        phy_NN_res_pivot = full_pivot_table(heatmap_df, "phy_NN_res")

        sns.heatmap(phy_res_pivot, cmap='seismic', center=0, cbar_kws={'pad': 0.02}, vmin=minimum, vmax=maximum,
                    ax=ax1).set_facecolor('#CCCCCC')
        sns.heatmap(phy_NN_res_pivot, cmap='seismic', center=0, cbar_kws={'pad': 0.02}, vmin=minimum, vmax=maximum,
                    ax=ax2).set_facecolor('#CCCCCC')

    ax1.set_title('Physical Model', pad=15, fontsize=25)
    ax2.set_title(f"Physical Model + Neural Network ({model.__class__.__name__})", pad=15, fontsize=25)

    for ax in [ax1, ax2]:
        set_chart_style(ax)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel('N', fontsize=25)
        ax.set_ylabel('Z', fontsize=25)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=10)
        if scale:
            colorbar.set_label("Normalized Residual", fontsize=20)
        else:
            colorbar.set_label("Residual(MeV)", fontsize=20)

    plt.show()


def plot_isotopic_res(model, Z, input_neurons):
    all_dataset_path = "data/all_dataset.csv"
    eva_nuclides = pd.read_csv(all_dataset_path)

    isotopes = eva_nuclides[eva_nuclides["Z"] == Z]

    train_dataset_path = "data/train_dataset.csv"
    train_nuclides = pd.read_csv(train_dataset_path)

    input_data = isotopes[input_neurons].values
    input_data = torch.tensor(input_data, dtype=torch.float32).to(cf.device)

    key_columns = ["Z", "N"]
    N_in_train = pd.merge(isotopes, train_nuclides, on=key_columns, how="inner")
    N_not_in_train = isotopes[~isotopes["N"].isin(N_in_train["N"])]

    model.eval()
    with torch.no_grad():
        output_data = model(input_data).cpu().numpy()

    plot_data = isotopes[["Z", "N"]].copy()
    plot_data["LDM_residual(MeV)"] = isotopes["LDM_residual(MeV)"]
    plot_data["NN_pred"] = output_data
    plot_data["refined_LDM"] = isotopes["LDM_residual(MeV)"] - plot_data["NN_pred"]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_data["N"], plot_data["refined_LDM"], label="LDM+NN", marker='o')
    plt.plot(plot_data["N"], plot_data["LDM_residual(MeV)"], label="LDM", marker='s')
    plt.axhline(0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel("N", fontsize=16)
    plt.ylabel("Residual (MeV)", fontsize=16)
    plt.title(f"Residuals in the Z = {Z} isotopic chain", fontsize=16)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(1)
        ax.spines[spine].set_edgecolor('black')

    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True, labelleft=True,
                   labelright=False, labeltop=False, direction='in', length=6, width=1)
    ax.tick_params(which='minor', length=3, width=1)

    for n in N_not_in_train["N"]:
        ax.axvspan(n - 0.5, n + 0.5, color='gray', alpha=0.3, hatch='/')

    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=14)
    plt.show()
