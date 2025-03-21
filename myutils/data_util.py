import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from config import TRAIN_DATA_PATH
from myutils.plot_util import plot_sample_nuclides_distribution


class Data(Dataset):
    def __init__(self, data_features, data_labels, feature_names=None):
        self.data_features = data_features
        self.data_labels = data_labels
        self.feature_names = feature_names

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, index):
        return self.data_features[index], self.data_labels[index]


def data_read(features, labels, filepath=TRAIN_DATA_PATH):
    """
    Reads data from a CSV file and returns a PyTorch Dataset.

    Parameters:
    filepath (str): Path to the CSV file.
    features (list): List of feature column names.
    labels (str): List of label column names.

    Returns:
    dataset: A PyTorch Dataset containing the features and labels.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_data = pd.read_csv(filepath)

    x = raw_data[features].values
    y = raw_data[labels].values
    feature_names = features

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    dataset = Data(x_tensor, y_tensor, feature_names)

    print(f'Number of samples in the entire dataset: {len(dataset)}\n'
          f'The third one, as an example, is:\n{dataset.__getitem__(2)}')

    return dataset


def proton_1_separation_energy(dataframe, head_of_BindingEnergy):

    binding_energies = {}
    for index, row in dataframe.iterrows():
        Z = row['Z']
        N = row['N']
        binding_energy = row[head_of_BindingEnergy]
        binding_energies[(Z, N)] = binding_energy

    Sp = []
    for (Z, N) in binding_energies:
        if (Z, N) in binding_energies and (Z - 1, N) in binding_energies:
            Sp.append(binding_energies[(Z, N)] - binding_energies[(Z - 1, N)])
        else:
            Sp.append(None)
    return Sp


def proton_2_separation_energy(dataframe, head_of_BindingEnergy):

    binding_energies = {(row['Z'], row['N']): row[head_of_BindingEnergy] for index, row in dataframe.iterrows()}

    S2p = [
        binding_energies[(Z, N)] - binding_energies[(Z - 2, N)]
        if (Z, N) in binding_energies and (Z - 2, N) in binding_energies
        else None
        for (Z, N) in binding_energies
    ]
    return S2p


def neutron_1_separation_energy(dataframe, head_of_BindingEnergy):

    # creates a dictionary of binding energies {key:(Z, N): value:binding energy}
    binding_energies = {(row['Z'], row['N']): row[head_of_BindingEnergy] for index, row in dataframe.iterrows()}

    Sn = [
        binding_energies[(Z, N)] - binding_energies[(Z, N - 1)]
        if (Z, N) in binding_energies and (Z, N - 1) in binding_energies
        else None
        for (Z, N) in binding_energies
    ]
    return Sn


def neutron_2_separation_energy(dataframe, head_of_BindingEnergy):

    binding_energies = {}
    for index, row in dataframe.iterrows():
        Z = row['Z']
        N = row['N']
        binding_energy = row[head_of_BindingEnergy]
        binding_energies[(Z, N)] = binding_energy

    S2n = []
    for (Z, N) in binding_energies:
        if (Z, N) in binding_energies and (Z, N - 2) in binding_energies:
            S2n.append(binding_energies[(Z, N)] - binding_energies[(Z, N - 2)])
        else:
            S2n.append(None)
    return S2n


def split_and_load(dataset, random_seed, train_proportion):
    """
    This function has three functionalities:
        Splits the dataset into training and testing sets with a specified random seed,
        Load the training and testing sets into DataLoaders.
        Plots the distribution of nuclides.
    It returns the train_loader and test_loader.

    Parameters:
        dataset (Dataset): The full dataset to be split.
        random_seed (int): The seed for reproducibility.
        train_proportion (float): Proportion of the dataset used for training.

    Plots:
        sample_distribution: Sample Distribution in the Chart of Nuclides.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the testing set.
    """
    # Set the random seed for reproducibility for the dataset split
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # Split the dataset
    train_set, test_set = random_split(dataset, [train_proportion, 1 - train_proportion])

    # Set the random seed for reproducibility again for the DataLoader
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    # Extract Z (proton number) and N (neutron number) for both sets
    Z_train = [data[0][0].cpu().detach().numpy() for data in train_set]
    N_train = [data[0][1].cpu().detach().numpy() for data in train_set]
    Z_test = [data[0][0].cpu().detach().numpy() for data in test_set]
    N_test = [data[0][1].cpu().detach().numpy() for data in test_set]

    # Plot the distribution using the provided plot function
    sample_distribution = plot_sample_nuclides_distribution(N_train, Z_train, N_test, Z_test)

    print(f"After data split by the random seed {random_seed}\n"
          f"Number of samples in the training set: {train_loader.batch_size}\n"
          f"Number of samples in the test set: {test_loader.batch_size}\n"
          f"Number of batches in the train_loader: {len(train_loader)}\n"
          f"Number of batches in the test_loader: {len(test_loader)}\n"
          f"Sample distribution: {sample_distribution}")

    return train_loader, test_loader


def compute_nuclide_sep_rmsd(model, physical_model_name):
    train_dataset = pd.read_csv('data/train_dataset.csv')
    dataset = train_dataset[['N', 'Z', 'B_exp(MeV)', f'B_{physical_model_name}(MeV)']].copy()

    res_pred = pd.read_csv(f"model/prediction/pred_{model.__class__.__name__}.csv")
    dataset[f"B_{physical_model_name}_NN(MeV)"] = dataset[f"B_{physical_model_name}(MeV)"] + res_pred['pred_NN']

    sep_funcs = {
        "Sn": neutron_1_separation_energy,
        "S2n": neutron_2_separation_energy,
        "Sp": proton_1_separation_energy,
        "S2p": proton_2_separation_energy,
    }

    data_groups = [
        ("exp", "B_exp(MeV)"),
        (physical_model_name, f"B_{physical_model_name}(MeV)"),
        (f"{physical_model_name}_NN", f"B_{physical_model_name}_NN(MeV)")
    ]

    for tag, head in data_groups:
        for key, func in sep_funcs.items():
            col_name = f"{key}_{tag}(MeV)"
            dataset[col_name] = func(dataframe=dataset, head_of_BindingEnergy=head)

    performance = {}
    for tag in [physical_model_name, f"{physical_model_name}_NN"]:
        performance[tag] = {}
        for key in sep_funcs.keys():
            exp_col = f"{key}_exp(MeV)"
            model_col = f"{key}_{tag}(MeV)"
            residual = dataset[exp_col] - dataset[model_col]
            rmsd = (residual ** 2).mean() ** 0.5
            performance[tag][key] = rmsd

    print(f"Physical model: {physical_model_name}")
    print(f"NN model: {model.__class__.__name__}")
    print(performance)
    return performance
