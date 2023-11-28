from utils import find_change_indices
import numpy as np
from ruptures.metrics.precisionrecall import precision_recall
from sklearn.model_selection import KFold
import ruptures as rpt
from typing import List, Union, Tuple, Any


def f1_score(
    predictions: List[List[Union[int, float]]], np_true_segmentations: np.ndarray, marge : float = (5 / 100)
) -> List[Union[float, float]]:
    """
    Compute F1 scores for a list of predicted segmentations.

    Args:
        predictions (List[List[Union[int, float]]]): A list of predicted segmentations as lists.
        np_true_segmentations (np.ndarray): The true segmentations data as a numpy array.

    Returns:
        List[Union[float, float]]: A list containing:
            - Mean F1 score (float) computed across predictions.
            - Standard deviation of F1 scores (float).
    """
    true_segmentations = [
        find_change_indices(np_true_segmentations[i])
        for i in range(len(np_true_segmentations))
    ]

    def solo_f1_score(
        predicted_segmentation: List[Union[int, float]], true_segmentation: List[int]
    ) -> float:
        precision, recall = precision_recall(
            np.array(predicted_segmentation),
            np.array(true_segmentation),
            margin=np_true_segmentations.shape[1] * marge,
        )
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    f1_scores = []
    for i in range(len(predictions)):
        f1_scores.append(solo_f1_score(true_segmentations[i], predictions[i]))

    return f1_scores


def crossvalidate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    num_folds: int = 5,
    num_epochs: int = 10,
    batch_size: int = 5,
    verbose: bool = False,
    marges : List[float] = [5/100, 4/100, 3/100, 2/100, 1/100]
) -> List[Tuple[float, float]]:
    """
    Perform k-fold cross-validation for a given model.

    Args:
        model (Any): The machine learning model to be evaluated.
        X (np.ndarray): The input features as a numpy array.
        y (np.ndarray): The target labels as a numpy array.
        num_folds (int, optional): Number of folds for cross-validation.
        num_epochs (int, optional): Number of training epochs.
        batch_size (int, optional): Batch size for training.
        verbose (bool, optional): Whether to print training progress.

    Returns:
        List[Tuple[float, float]]: A list of tuples containing:
            - Mean Test Score (float) across folds.
            - Standard Deviation of Test Scores (float) across folds.
    """
    print(
        "Starting Cross Validation for "
        + str(num_folds)
        + " splits and "
        + str(num_epochs)
        + " epochs. (batch size : "
        + str(batch_size)
        + ")"
    )
    print("...")
    # Initialize lists to store scores for each fold
    test_scores = np.zeros((len(marges), len(X)))

    # Create a KFold object
    kf = KFold(
        n_splits=num_folds, shuffle=True, random_state=42
    )  # You can adjust 'shuffle' and 'random_state'

    # Iterate over the folds
    last_indx = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        model.params = model.initialize_parameters_method(
            model.parameters_informations, verbose=False
        )

        # Train your model on X_train and y_train
        # Replace 'YourModel' with your actual machine learning model
        model.fit(
            X_train,
            y_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        j = 0
        for marge in marges:
            # Calculate and store the test scores
            test_score = f1_score(y_pred, y_test, marge)
            test_scores[j][last_indx:last_indx + len(y_test)] = np.array(test_score)
            j+=1
        last_indx = last_indx + len(y_test)

    print(test_scores)
    # Calculate the mean and standard deviation of test scores
    test_scores = np.array(test_scores)
    mean_test_score = np.mean(test_scores, axis = 1)
    std_test_score = np.std(test_scores, axis = 1)

    # Print or analyze the results as needed
    print("Cross Validation results :")
    print('Margins :', [marge * y.shape[1] for marge in marges])
    print("Mean Test Score:", mean_test_score)
    print("Standard Deviation of Test Scores:", std_test_score)

    return test_scores


def bic(signal_length: int, sigma: float, n_dims: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC).

    Args:
        signal_length (int): The length of the signal.
        sigma (float): The standard deviation parameter.
        n_dims (int): The number of dimensions.

    Returns:
        float: The BIC value.
    """
    return 2 * np.log(signal_length) * (sigma**2) * n_dims


def give_bic_acc(
    signals: List[np.ndarray],
    true_segmentations: List[np.ndarray],
    signal_length: int,
    sigma: float,
    n_dims: int,
) -> List[Union[float, float]]:
    """
    Compute BIC-based accuracy scores for a list of signals and true segmentations.

    Args:
        signals (List[np.ndarray]): A list of input signals as numpy arrays.
        true_segmentations (List[np.ndarray]): A list of true segmentation data as numpy arrays.
        signal_length (int): The length of the signal.
        sigma (float): The standard deviation parameter.
        n_dims (int): The number of dimensions.

    Returns:
        List[Union[float, float]]: A list containing:
            - Mean F1 score (float) across signals.
            - Standard Deviation of F1 scores (float) across signals.
    """
    f1_scores = []

    for i in range(len(signals)):
        algo = rpt.KernelCPD(kernel="linear", min_size=3, jump=1).fit(signals[i])
        bic_pen = bic(signal_length, sigma, n_dims)
        predicted_segmentation_i = algo.predict(pen=bic_pen)

        precision, recall = precision_recall(
            np.array(find_change_indices(true_segmentations[i])),
            predicted_segmentation_i,
            margin=signals[0].shape[0] * (5 / 100),
        )
        if precision + recall == 0:
            score = 0
        else:
            score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(score)

    f1_scores_mean = np.mean(f1_scores)

    return [f1_scores_mean, np.std(f1_scores)]


def crossvalidate_class(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    true_classification: np.ndarray,
    num_folds: int = 5,
    num_epochs: int = 10,
    batch_size: int = 5,
    verbose: bool = False,
    marges : List[float] = [5/100, 4/100, 3/100, 2/100, 1/100]
) -> List[Tuple[float, float]]:
    """
    Perform k-fold cross-validation for a given model.

    Args:
        model (Any): The machine learning model to be evaluated.
        X (np.ndarray): The input features as a numpy array.
        y (np.ndarray): The target labels as a numpy array.
        num_folds (int, optional): Number of folds for cross-validation.
        num_epochs (int, optional): Number of training epochs.
        batch_size (int, optional): Batch size for training.
        verbose (bool, optional): Whether to print training progress.

    Returns:
        List[Tuple[float, float]]: A list of tuples containing:
            - Mean Test Score (float) across folds.
            - Standard Deviation of Test Scores (float) across folds.
    """
    print(
        "Starting Cross Validation for "
        + str(num_folds)
        + " splits and "
        + str(num_epochs)
        + " epochs. (batch size : "
        + str(batch_size)
        + ")"
    )
    print("...")
    # Initialize lists to store scores for each fold
    test_scores = np.zeros((len(marges), len(X)))

    # Create a KFold object
    kf = KFold(
        n_splits=num_folds, shuffle=True, random_state=42
    )  # You can adjust 'shuffle' and 'random_state'

    # Iterate over the folds
    last_indx = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        true_class_train, true_class_test = true_classification[train_index], true_classification[test_index]
        model.params = model.initialize_parameters_method(
            model.parameters_informations, verbose=False
        )
        # Train your model on X_train and y_train
        # Replace 'YourModel' with your actual machine learning model
        model.fit(
            X_train,
            y_train,
            true_class_train,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        j = 0
        for marge in marges:
            # Calculate and store the test scores
            test_score = f1_score(y_pred, y_test, marge)
            test_scores[j][last_indx:last_indx + len(y_test)] = np.array(test_score)
            j+=1
        last_indx = last_indx + len(y_test)

    print(test_scores)
    # Calculate the mean and standard deviation of test scores
    test_scores = np.array(test_scores)
    mean_test_score = np.mean(test_scores, axis = 1)
    std_test_score = np.std(test_scores, axis = 1)

    # Print or analyze the results as needed
    print("Cross Validation results :")
    print('Margins :', [marge * y.shape[1] for marge in marges])
    print("Mean Test Score:", mean_test_score)
    print("Standard Deviation of Test Scores:", std_test_score)

    return test_scores