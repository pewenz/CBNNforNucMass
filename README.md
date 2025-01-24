# Deep Learning for Nuclear Mass Prediction

## Description
This project focuses on training neural networks to predict the residuals (in MeV) between theoretical models and experimental data, with a variety of nuclear features as inputs.

Using a custom architecture from the repository and trained on the AME2020 dataset, the NN models make predictions based on the trained parameters.

You can play with this project by modifying the features, labels, model architecture, optimizer, etc. to see how the model performance changes.

## Table of Contents
- [Data Part](#data-part)
- [Model Part](#model-part)
- [License](#license)

## Data Part

### Steps:
1. **Feature and Label Selection**: Choose the nuclear features and labels for the model.
2. **Data Splitting**: Set the random seed and proportion for splitting the data.
3. **Data Reading**: Read the dataset from the file.
4. **Data Splitting**: Split and load the data into training and testing data loaders.
5. **Data Distribution Plotting**: Plot the data distribution for visual analysis.

### Example Code:
```python
features = ['Z', 'N', 'N-Z', 'A', 'P', 'D']
labels = ['LDM_residual(MeV)']
random_seed = 39
train_proportion = 0.8

dataset = data_read(features=features, labels=labels)
train_loader, test_loader = split_and_load(dataset=dataset, random_seed=random_seed, train_proportion=train_proportion)
```

### Configuration:

- **Features**: `['Z', 'N', 'N-Z', 'A', 'P', 'D']`
- **Labels**: `['LDM_residual(MeV)']`
- **Random Seed**: `39`
- **Training Proportion**: `0.8` (80% for training, 20% for testing)

## Model Part

### Steps:
1. **Model Architecture**: Build your own model architecture in the repository `model_repo.py`.
2. **Optimizer and Scheduler**: Set up the optimizer, scheduler, etc.
3. **Model Training**: Train the model with the training data and test it with the test data.
4. **Monitoring**: Track the learning process by plotting the learning curve.
5. **Prediction**: Use the trained model to make predictions.

### Training Example:
```python
train_model(epochs_add=5000, 
            model=model, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            optimizer=optimizer, 
            monitor_interval=1000,
            scheduler=scheduler,
            use_early_stopping=True,
            patience=10)
```

`epochs_add`: Number of additional epochs to train the model.

`monitor_interval`: Interval for monitoring the training process, also the space between two checkpoints in the learning curve

`patience`: If no improvement is observed after a certain number of epochs, the training process would be stopped to avoid overfitting.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
more details on specific parts of the README or include other sections?