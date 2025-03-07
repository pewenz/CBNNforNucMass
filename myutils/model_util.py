import torch
import torch.nn as nn
import os
import pandas as pd
import re
from myutils.plot_util import plot_learning
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def msrd(pre, tar, eps=1e-8):
    """Calculate Mean Squared Relative Deviation (MSRD)."""
    return ((pre - tar) / (tar + eps)).pow(2).mean()


def get_epochs_done(record_path):
    """Read the number of epochs already trained from the record file."""
    if not os.path.exists(record_path):
        return 0
    with open(record_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return 0
        last_line = lines[-1]
        if "Epoch" in last_line:
            return int(last_line.split()[1].strip(',').strip('[]'))
        return 0


def get_lowest_test_rmsd(record_path):
    """
    Reads the lowest test RMSD from the record file.

    Parameters:
    record_path (str): Path to the record file.

    Returns:
    float: The lowest test RMSD value. If the file doesn't exist or no valid RMSD is found, returns float('inf').
    """
    # Check if the file exists
    if not os.path.exists(record_path):
        return float('inf')

    lowest_test_rmsd = float('inf')

    try:
        # Open and process the file
        with open(record_path, 'r') as f:
            for line in f:
                # Check if the line contains "Test RMSD"
                if "Test RMSD" in line:
                    try:
                        # Extract the last numeric value in the line
                        test_rmsd = float(line.split()[5])
                        # Update the lowest RMSD if the current one is smaller
                        lowest_test_rmsd = min(lowest_test_rmsd, test_rmsd)
                    except ValueError:
                        # Handle cases where conversion to float fails
                        continue
    except IOError as e:
        # Handle potential file I/O errors
        print(f"Error reading the file {record_path}: {e}")
        return float('inf')

    return lowest_test_rmsd


def early_stopping(patience, best_parameter_path, test_RMSD, lowest_test_RMSD, intervals_no_improve, epoch_now, model):
    """
    Early stopping mechanism to save the best model and stop training if no improvement.
    Parameters:
    - patience: int, Number of epochs to wait before stopping.
    - best_parameter_path: str, Path to save the best model parameters.
    - test_RMSD: float, Current test RMSD.
    - epochs_no_improve: int, Number of epochs with no improvement.
    - model: torch.nn.Module, The PyTorch model to save.
    Returns:
    - best_test_RMSD: float, Updated best test RMSD.
    - epochs_no_improve: int, Updated number of epochs with no improvement.
    - stop_training: bool, Whether to stop training.
    """
    stop_training = False

    if test_RMSD <= lowest_test_RMSD:
        lowest_test_RMSD = test_RMSD
        intervals_no_improve = 0
        torch.save(model.state_dict(), best_parameter_path)
        print(f"Epoch [{epoch_now}], Test RMSD improved to {lowest_test_RMSD:.6f}, better model parameters saved to `{best_parameter_path}`\n")
    else:
        print(f"Epoch [{epoch_now}], Test RMSD = {test_RMSD:.6f} doesn't improve, lowest test RMSD = {lowest_test_RMSD:.6f}\n")
        intervals_no_improve += 1

    if intervals_no_improve >= patience:
        stop_training = True

    return stop_training, intervals_no_improve


def train_model(epochs_add, model, train_loader, test_loader, optimizer, monitor_interval, scheduler=None, use_early_stopping=False, patience=5):
    """
    Train and evaluate the NucB_model.
    Parameters:
    - epochs_add: int, Number of training epochs this time.
    - model: torch.nn.Module, The PyTorch model to train.
    - train_loader: torch.utils.data.DataLoader, DataLoader for training data.
    - test_loader: torch.utils.data.DataLoader, DataLoader for testing data.
    - optimizer: torch.optim.Optimizer, Optimizer for the model.
    - monitor_interval: int, Interval for monitoring and recording training/testing results.
    - scheduler: torch.optim.lr_scheduler, Learning rate scheduler.
    - use_early_stopping: bool, Whether to use early stopping mechanism.
    - patience: int, Number of epochs to wait for improvement before stopping.
    """
    model = model.to(device)
    if len(train_loader) == 0 or len(test_loader) == 0:
        raise ValueError("train_loader and test_loader must not be empty.")

    record_path = f"model/train_record/record_{model.__class__.__name__}.txt"
    best_parameter_path = f"model/parameter/param_{model.__class__.__name__}.pth"

    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_parameter_path), exist_ok=True)

    epochs_done = get_epochs_done(record_path)
    lowest_test_RMSD = get_lowest_test_rmsd(record_path)
    print(f"Model has already been trained for {epochs_done} epochs, and the lowest test RMSD = {lowest_test_RMSD}\n")

    if os.path.exists(best_parameter_path):
        model.load_state_dict(torch.load(best_parameter_path))
        print(f"Optimal parameters loaded from `{best_parameter_path}`\n")

    print(f"Now, training on {device} for {epochs_add} epochs...\n"
          f"Training process will be recorded in `{record_path}`\n")

    msd_loss = nn.MSELoss()
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs_add)):

        train_MSD = 0.0
        train_MSRD = 0.0

        # Training phase
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            targets = targets.squeeze()

            outputs = model(inputs).squeeze()

            batch_MSD = msd_loss(outputs, targets)
            batch_MSRD = msrd(outputs, targets)

            optimizer.zero_grad()
            batch_MSD.backward()
            optimizer.step()

            train_MSD += batch_MSD.item()
            train_MSRD += batch_MSRD.item()

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Print results at specified intervals
        if (epoch + 1) % monitor_interval == 0:
            train_RMSD = (train_MSD / len(train_loader)) ** (1 / 2)
            train_RMSRD = (train_MSRD / len(train_loader)) ** (1 / 2)

            with open(record_path, 'a') as f:
                f.write(
                    f"Epoch [{epochs_done + epoch + 1}], Train RMSD = {train_RMSD:.6f} (MeV), Train RMSRD = {train_RMSRD:.6f}, LR = {current_lr}\n")

            # Testing phase
            model.eval()
            with torch.inference_mode():
                test_MSD = 0.0
                test_MSRD = 0.0

                for test_features, test_targets in test_loader:
                    test_targets = test_targets.squeeze()

                    test_output = model(test_features).squeeze()

                    test_MSD += msd_loss(test_output, test_targets).item()
                    test_MSRD += msrd(test_output, test_targets).item()

                test_RMSD = (test_MSD / len(test_loader)) ** (1 / 2)
                test_RMSRD = (test_MSRD / len(test_loader)) ** (1 / 2)

                if test_RMSD < lowest_test_RMSD:
                    lowest_test_RMSD = test_RMSD

                with open(record_path, 'a') as f:
                    f.write(
                        f"Epoch [{epochs_done + epoch + 1}], Test RMSD = {test_RMSD:.6f} (MeV), Test RMSRD = {test_RMSRD:.6f}\n")

                if use_early_stopping:
                    stop_training, epochs_no_improve = early_stopping(patience, best_parameter_path, test_RMSD, lowest_test_RMSD, epochs_no_improve, epochs_done + epoch + 1, model)
                    if stop_training:
                        print(f"Training stopped early due to no improvement after {epochs_no_improve * monitor_interval} epochs.\n")
                        break


def monitor_learning(model, train_proportion, find_optimal=True, lr_changes_at=None):
    if lr_changes_at is None:
        lr_changes_at = []

    record_path = f"model/train_record/record_{model.__class__.__name__}.txt"
    record = pd.read_fwf(record_path, header=None, delimiter='\n')

    train_pattern = r"Epoch \[(\d+)\], Train RMSD = ([\d\.]+) \(MeV\), Train RMSRD = ([\d\.]+), LR = ([\d\.]+)"
    test_pattern = r"Epoch \[(\d+)\], Test RMSD = ([\d\.]+) \(MeV\), Test RMSRD = ([\d\.]+)"

    epochs = []
    train_rmsd = []
    train_rmsrd = []
    test_rmsd = []
    test_rmsrd = []

    for line in record[0]:
        train_match = re.match(train_pattern, line)
        test_match = re.match(test_pattern, line)
        if train_match:
            epochs.append(int(train_match.group(1)))
            train_rmsd.append(float(train_match.group(2)))
            train_rmsrd.append(float(train_match.group(3)))
        elif test_match:
            test_rmsd.append(float(test_match.group(2)))
            test_rmsrd.append(float(test_match.group(3)))

    learning_curve = plot_learning(train_proportion, find_optimal, epochs, train_rmsd, test_rmsd, train_rmsrd,
                                   test_rmsrd, lr_changes_at)

    return learning_curve


def model_predict(model, dataset):
    """
    Predicts the output of a model on a dataset.

    Parameters:
    model: A PyTorch model.
    dataset: A PyTorch Dataset.

    Returns:
    prediction: A NumPy array containing the model's predictions.
    """
    model.eval()
    with torch.inference_mode():
        prediction = model(dataset.data_features).squeeze()

    prediction = prediction.squeeze().cpu().numpy()

    features_df = pd.DataFrame(dataset.data_features.cpu().numpy())

    pred_df = pd.DataFrame({
        'Z': features_df[features_df.columns[dataset.feature_names.index('Z')]].astype(int),
        'N': features_df[features_df.columns[dataset.feature_names.index('N')]].astype(int),
        'pred_NN': prediction
    })

    pred_path = f"model/prediction/pred_{model.__class__.__name__}.csv"

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pred_df.to_csv(pred_path, index=False, header=True)

    print(f"{len(prediction)} predictions saved at `{pred_path}`")
