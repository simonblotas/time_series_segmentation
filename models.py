from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
import optax
import ruptures as rpt
import jax.numpy as jnp
from jax import vmap
import time
from breakpoints_computation import get_optimal_projection
from utils import find_change_indices, create_data_loader, create_data_loader_classification
from loss_related_functions import final_loss_and_grad, final_loss_and_grad_autoencoder, final_loss_and_grad_bin, final_loss_and_grad_classification, compute_cross_entropy_loss, final_loss_and_grad_classification_bin
from default_optimizer import gradient_transform
from ruptures.metrics.precisionrecall import precision_recall
from scipy.signal import find_peaks


class SimplePipeline(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        transformation_method,
        initialize_parameters_method,
        parameters_informations,
        optimizer=gradient_transform,
        
    ):
        self.parameters_informations = parameters_informations
        self.initialize_parameters_method = initialize_parameters_method
        self.params = self.initialize_parameters_method(
            self.parameters_informations, verbose=True
        )
        self.transformation_method = transformation_method
        self.optimizer = optimizer
        

    def fit(
        self,
        signals,
        segmentations,
        verbose=False,
        num_epochs=10,
        batch_size=5,
        test_batch_idx=[],
        static_params={},
        **fit_params
    ):
        """Implements a learning loop over epochs."""

        # Initialize static parameters
        for key, value in static_params.items():
            self.params[key] = value

        # Rest of the function remains the same
        train_loss = []
        acc_train = []
        acc_test = []
        # Initialize optimizer.

        opt_state = self.optimizer.init(self.params)

        # Loop over the training epochs

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loader = create_data_loader(
                signals, segmentations, batch_size, test_batch_idx
            )

            bacth_losses = []
            batch_acc_train = []
            batch_acc_test = []

            for batch_type, (data, target) in train_loader:
                if batch_type == "Train":
                    data = jnp.array(data)
                    target = jnp.array(target)
                    # Compute the loss and gradients using only dynamic parameters
                    value, grads = final_loss_and_grad(
                        self.params, self.transformation_method, data, target
                    )
                    # Update only the dynamic parameters
                    updates, opt_state = self.optimizer.update(grads, opt_state)
                    self.params = optax.apply_updates(self.params, updates)

                    for key, value in static_params.items():
                        self.params[key] = value

                    bacth_losses.append(value)

                elif batch_type == "Test":
                    pass

            epoch_loss = np.mean(bacth_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time

            if verbose:
                print(
                    "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                        epoch + 1,
                        epoch_time,
                        epoch_acc_train,
                        epoch_acc_test,
                        epoch_loss,
                        jnp.exp(self.params["beta"]),
                    )
                )

        return train_loss, opt_state, self.params

    def predict(self, signals):
        def predict_segmentation(self, signal: jnp.ndarray) -> jnp.ndarray:
            """
            Predicts the segmentation of a given signal using the trained scaling network.

            Parameters:
            signal (jnp.ndarray): The input signal for segmentation prediction.

            Returns:
            jnp.ndarray: Predicted segmentation indices for the input signal.
            """
            # Transform the input signal using the network's transformation function
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signal)
            )

            # Convert the transformed signal to a numpy array for compatibility
            # transformed_signal_array = np.array(transformed_signal)

            # Get the optimal projection and predicted segmentation using the transformed signal
            (
                pred_projection,
                pred_segmentation_size,
                segment_ids_pred,
            ) = get_optimal_projection(
                transformed_signal, penalty=jnp.exp(self.params["beta"])
            )
            # print(segment_ids_pred[1])
            # Find and return the predicted segmentation indices
            # predicted_segmentation = find_change_indices(segment_ids_pred[1])
            return segment_ids_pred[1]
            # Make a batched version of the `predict_segmentation` function

        jnp_predictions = vmap(predict_segmentation, in_axes=(None, 0), out_axes=0)(
            self, signals
        )
        predictions = [
            find_change_indices(jnp_predictions[i]) for i in range(len(jnp_predictions))
        ]

        return predictions

    def display(self, signals, true_segmentations):
        predictions = self.predict(signals)

        print("Used margin = ", true_segmentations.shape[1] * (5 / 100))
        print("--------------------------------------")
        for i in range(len(signals)):
            print("Signal n° ", i)
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signals[i])
            )
            precision, recall = precision_recall(
                np.array(find_change_indices(true_segmentations[i])),
                np.array(predictions[i]),
                margin=true_segmentations.shape[1] * (5 / 100),
            )
            
            print("True segmentation :", find_change_indices(true_segmentations[i]))
            print("Predicted segmentation :", predictions[i])
            if precision + recall == 0:
                print("F1 score :", 0)
            else:
                print("F1 score :", 2 * (precision * recall) / (precision + recall))
            rpt.display(
                np.array(transformed_signal),
                find_change_indices(true_segmentations[i]),
                predictions[i],
            )
            plt.show()


class PipelineAutoencoder(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        encoder_method,
        decoder_method,
        initialize_parameters_method,
        parameters_informations,
        optimizer=gradient_transform,
        lambda_reconstruction = 1.,
        lambda_segmentation = 1.,
    ):
        self.parameters_informations = parameters_informations
        self.initialize_parameters_method = initialize_parameters_method
        self.params = self.initialize_parameters_method(
            self.parameters_informations, verbose=True
        )
        self.encoder = encoder_method
        self.decoder = decoder_method
        self.optimizer = optimizer
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_segmentation = lambda_segmentation

    def fit(
        self,
        signals,
        segmentations,
        verbose=False,
        num_epochs=10,
        batch_size=5,
        test_batch_idx=[],
        static_params={},
        **fit_params
    ):
        """Implements a learning loop over epochs."""

        # Initialize static parameters
        for key, value in static_params.items():
            self.params[key] = value

        # Rest of the function remains the same
        train_loss = []
        acc_train = []
        acc_test = []
        # Initialize optimizer.

        opt_state = self.optimizer.init(self.params)

        # Loop over the training epochs

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loader = create_data_loader(
                signals, segmentations, batch_size, test_batch_idx
            )

            bacth_losses = []
            batch_acc_train = []
            batch_acc_test = []

            for batch_type, (data, target) in train_loader:
                if batch_type == "Train":
                    data = jnp.array(data)
                    target = jnp.array(target)
                    # Compute the loss and gradients using only dynamic parameters
                    value, grads = final_loss_and_grad_autoencoder(
                        self.params, self.encoder, self.decoder, data, target, self.lambda_reconstruction, self.lambda_segmentation
                    )
                    # Update only the dynamic parameters
                    updates, opt_state = self.optimizer.update(grads, opt_state)
                    self.params = optax.apply_updates(self.params, updates)

                    for key, value in static_params.items():
                        self.params[key] = value

                    bacth_losses.append(value)

                elif batch_type == "Test":
                    pass

            epoch_loss = np.mean(bacth_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time

            if verbose:
                print(
                    "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                        epoch + 1,
                        epoch_time,
                        epoch_acc_train,
                        epoch_acc_test,
                        epoch_loss,
                        jnp.exp(self.params["beta"]),
                    )
                )

        return train_loss, opt_state, self.params

    def predict(self, signals):
        def predict_segmentation(self, signal: jnp.ndarray) -> jnp.ndarray:
            """
            Predicts the segmentation of a given signal using the trained scaling network.

            Parameters:
            signal (jnp.ndarray): The input signal for segmentation prediction.

            Returns:
            jnp.ndarray: Predicted segmentation indices for the input signal.
            """
            # Transform the input signal using the network's transformation function
            transformed_signal = self.encoder(
                self.params, jnp.array(signal)
            )

            # Convert the transformed signal to a numpy array for compatibility
            # transformed_signal_array = np.array(transformed_signal)

            # Get the optimal projection and predicted segmentation using the transformed signal
            (
                pred_projection,
                pred_segmentation_size,
                segment_ids_pred,
            ) = get_optimal_projection(
                transformed_signal, penalty=jnp.exp(self.params["beta"])
            )
            # print(segment_ids_pred[1])
            # Find and return the predicted segmentation indices
            # predicted_segmentation = find_change_indices(segment_ids_pred[1])
            return segment_ids_pred[1]
            # Make a batched version of the `predict_segmentation` function

        jnp_predictions = vmap(predict_segmentation, in_axes=(None, 0), out_axes=0)(
            self, signals
        )
        predictions = [
            find_change_indices(jnp_predictions[i]) for i in range(len(jnp_predictions))
        ]

        return predictions

    def display(self, signals, true_segmentations):
        predictions = self.predict(signals)
        print("marge = ", signals.shape[1] * (5 / 100))
        for i in range(len(signals)):
            transformed_signal = self.encoder(
                self.params, jnp.array(signals[i])
            )
            precision, recall = precision_recall(
                np.array(find_change_indices(true_segmentations[i])),
                np.array(predictions[i]),
                margin=signals.shape[1] * (5 / 100),
            )
            print(predictions[i])
            print(find_change_indices(true_segmentations[i]))
            if precision + recall == 0:
                print(0)
            else:
                print(2 * (precision * recall) / (precision + recall))
            rpt.display(
                np.array(transformed_signal),
                find_change_indices(true_segmentations[i]),
                predictions[i],
            )
            plt.show()

class SimplePipelineBin(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        transformation_method,
        initialize_parameters_method,
        parameters_informations,
        optimizer=gradient_transform,
        alpha = 1,
    ):
        self.parameters_informations = parameters_informations
        self.initialize_parameters_method = initialize_parameters_method
        self.params = self.initialize_parameters_method(
            self.parameters_informations, verbose=True
        )
        self.transformation_method = transformation_method
        self.optimizer = optimizer
        self.alpha = alpha

    def fit(
        self,
        signals,
        segmentations,
        verbose=False,
        num_epochs=10,
        batch_size=5,
        test_batch_idx=[],
        **fit_params
    ):
        """Implements a learning loop over epochs."""
        y = np.zeros(segmentations.shape)
        for i in range(len(segmentations)) :
            change_indx = find_change_indices(segmentations[i])
            ground_truth = np.zeros(segmentations[i].shape)
            ground_truth[change_indx[:-1]] = 1
            y[i] = ground_truth

        self.params = self.initialize_parameters_method(self.parameters_informations)
        # Initialize placeholder fit data
        train_loss = []
        acc_train = []
        acc_test = []
        # Initialize optimizer.
        opt_state = self.optimizer.init(self.params)
        # Loop over the training epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loader = create_data_loader(
                signals, y, batch_size, test_batch_idx
            )
            bacth_losses = []
            batch_acc_train = []
            batch_acc_test = []
            for batch_type, (data, target) in train_loader:
                if batch_type == "Train":
                    data = jnp.array(data)
                    target = jnp.array(target)
                    value, grads = final_loss_and_grad_bin(
                        self.params, self.transformation_method, data, target, self.alpha
                    )
                    updates, opt_state = self.optimizer.update(grads, opt_state)
                    self.params = optax.apply_updates(self.params, updates)
                    bacth_losses.append(value)

                elif batch_type == "Test":
                    pass

            epoch_loss = np.mean(bacth_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time
            if verbose:
                print(
                    "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                        epoch + 1,
                        epoch_time,
                        epoch_acc_train,
                        epoch_acc_test,
                        epoch_loss,
                        jnp.exp(self.params["beta"]),
                    )
                )

        return train_loss, opt_state, self.params

    def predict(self, signals, height=0.5, distance = 50, threshold = 0., limit = 50.):
        predictions = []
        for i in range(len(signals)):
            transformed_signal = self.transformation_method(self.params, signals[i])
            peaks, _ = find_peaks(transformed_signal[:,0], height=height, distance = distance, threshold=threshold) 
            for element in peaks :
                if element < limit or element > transformed_signal.shape[0] - limit:
                    # Create a boolean mask to identify the element to delete
                    mask = peaks != element

                    # Use the mask to create a new array without the element to delete
                    peaks = peaks[mask]
    
            peaks = np.append(peaks, transformed_signal.shape[0])
            predictions.append(peaks)

        return predictions

    def display(self, signals, true_segmentations, height=0.5, distance = 50, threshold = 0., limit = 50.):
        predictions = []
        y = np.zeros(true_segmentations.shape)
        for i in range(len(true_segmentations)) :
            change_indx = find_change_indices(true_segmentations[i])
            ground_truth = np.zeros(true_segmentations[i].shape)
            ground_truth[change_indx[:-1]] = 1
            y[i] = ground_truth

        for i in range(len(signals)):
            transformed_signal = self.transformation_method(self.params, signals[i])
            peaks, _ = find_peaks(transformed_signal[:,0], height=height, distance = distance, threshold=threshold) 
            for element in peaks :
                if element < limit or element > transformed_signal.shape[0] - limit:
                    # Create a boolean mask to identify the element to delete
                    mask = peaks != element

                    # Use the mask to create a new array without the element to delete
                    peaks = peaks[mask]
    
            peaks = np.append(peaks, transformed_signal.shape[0])
            predictions.append(peaks)
        print("marge = ", signals.shape[1] * (5 / 100) )
        for i in range(len(signals)):
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signals[i])
            )
            print(predictions[i])
            print(find_change_indices(jnp.cumsum(y[i], axis=0, dtype=int)))
            if len(predictions[i]) > 1:
                precision, recall = precision_recall(
                np.array(predictions[i]),
                np.array(find_change_indices(jnp.cumsum(y[i], axis=0, dtype=int))),
                margin=signals.shape[1] * (5 / 100),
                )
                if precision + recall == 0:
                    print(0)
                else:
                    print(2 * (precision * recall) / (precision + recall))
            
            rpt.display(
                np.array(transformed_signal),
                find_change_indices(jnp.cumsum(y[i], axis=0, dtype=int)),
                predictions[i],
            )
            plt.show()


class SimplePipelineClassification(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        transformation_method,
        transformation_to_class_method,
        initialize_parameters_method,
        parameters_informations,
        optimizer=gradient_transform,
        lambda_classification = 1.,
        lambda_segmentation = 1.,
    ):
        self.parameters_informations = parameters_informations
        self.initialize_parameters_method = initialize_parameters_method
        self.params = self.initialize_parameters_method(
            self.parameters_informations, verbose=True
        )
        self.transformation_method = transformation_method
        self.transformation_to_class_method = transformation_to_class_method
        self.optimizer = optimizer
        self.lambda_classification = lambda_classification
        self.lambda_segmentation = lambda_segmentation

 
    def fit(
        self,
        signals,
        segmentations,
        true_classification,
        verbose=False,
        num_epochs=10,
        batch_size=5,
        test_batch_idx=[],
        static_params={},
        **fit_params
    ):
        """Implements a learning loop over epochs."""

        # Initialize static parameters
        for key, value in static_params.items():
            self.params[key] = value

        # Rest of the function remains the same
        train_loss = []
        acc_train = []
        acc_test = []
        # Initialize optimizer.

        opt_state = self.optimizer.init(self.params)

        # Loop over the training epochs

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loader = create_data_loader_classification(
                signals, segmentations,true_classification, batch_size, test_batch_idx
            )

            bacth_losses = []
            batch_acc_train = []
            batch_acc_test = []

            for batch_type, (data, target, true_classification_target) in train_loader:
                if batch_type == "Train":
                    data = jnp.array(data)
                    target = jnp.array(target)
                    true_classification_target = jnp.array(true_classification_target)
                    # Compute the loss and gradients using only dynamic parameters
                    value, grads = final_loss_and_grad_classification(
                        self.params, self.transformation_method, self.transformation_to_class_method, data, target, true_classification_target, self.lambda_classification, self.lambda_segmentation
                    )
                    # Update only the dynamic parameters
                    updates, opt_state = self.optimizer.update(grads, opt_state)
                    self.params = optax.apply_updates(self.params, updates)

                    for key, value in static_params.items():
                        self.params[key] = value

                    bacth_losses.append(value)

                elif batch_type == "Test":
                    pass

            epoch_loss = np.mean(bacth_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time

            if verbose:
                print(
                    "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                        epoch + 1,
                        epoch_time,
                        epoch_acc_train,
                        epoch_acc_test,
                        epoch_loss,
                        jnp.exp(self.params["beta"]),
                    )
                )

        return train_loss, opt_state, self.params

    def predict(self, signals):
        def predict_segmentation(self, signal: jnp.ndarray) -> jnp.ndarray:
            """
            Predicts the segmentation of a given signal using the trained scaling network.

            Parameters:
            signal (jnp.ndarray): The input signal for segmentation prediction.

            Returns:
            jnp.ndarray: Predicted segmentation indices for the input signal.
            """
            # Transform the input signal using the network's transformation function
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signal)
            )

            # Convert the transformed signal to a numpy array for compatibility
            # transformed_signal_array = np.array(transformed_signal)

            # Get the optimal projection and predicted segmentation using the transformed signal
            (
                pred_projection,
                pred_segmentation_size,
                segment_ids_pred,
            ) = get_optimal_projection(
                transformed_signal, penalty=jnp.exp(self.params["beta"])
            )
            # print(segment_ids_pred[1])
            # Find and return the predicted segmentation indices
            # predicted_segmentation = find_change_indices(segment_ids_pred[1])
            return segment_ids_pred[1]
            # Make a batched version of the `predict_segmentation` function

        jnp_predictions = vmap(predict_segmentation, in_axes=(None, 0), out_axes=0)(
            self, signals
        )
        predictions = [
            find_change_indices(jnp_predictions[i]) for i in range(len(jnp_predictions))
        ]

        return predictions

    def display(self, signals, true_segmentations, true_classification):
        predictions = self.predict(signals)
        print("Used margin = ", true_segmentations.shape[1] * (5 / 100))
        print("--------------------------------------")
        for i in range(len(signals)):
            print("Signal n° ", i)
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signals[i])
            )
            precision, recall = precision_recall(
                np.array(find_change_indices(true_segmentations[i])),
                np.array(predictions[i]),
                margin=true_segmentations.shape[1] * (5 / 100),
            )
            
            print("True segmentation :", find_change_indices(true_segmentations[i]))
            print("Predicted segmentation :", predictions[i])
            if precision + recall == 0:
                print("F1 score :", 0)
            else:
                print("F1 score :", 2 * (precision * recall) / (precision + recall))
            rpt.display(
                np.array(transformed_signal),
                find_change_indices(true_segmentations[i]),
                predictions[i],
            )
            plt.show()
            # Plotting side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            predicted_label = self.transformation_to_class_method(self.params,transformed_signal)
            print("Classification loss : " ,compute_cross_entropy_loss(predicted_label,true_classification[i]))

            # Plot predicted labels
            for j in range(true_classification.shape[2]):  # assuming 6 classes
                axes[0].plot(predicted_label[:, j], label=f'Class {j}')

            axes[0].set_title('Predicted Labels')
            axes[0].legend()
            axes[0].grid(True)

            # Plot true labels
            for j in range(true_classification.shape[2]):  # assuming 6 classes
                axes[1].plot(true_classification[i][:, j], label=f'Class {j}')

            axes[1].set_title('True Labels')
            axes[1].legend()
            axes[1].grid(True)

            plt.show()


class SimplePipelineClassificationBin(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        transformation_method,
        transformation_to_class_method,
        initialize_parameters_method,
        parameters_informations,
        optimizer=gradient_transform,
        lambda_classification = 1.,
        lambda_segmentation = 1.,
        alpha = 1,
    ):
        self.parameters_informations = parameters_informations
        self.initialize_parameters_method = initialize_parameters_method
        self.params = self.initialize_parameters_method(
            self.parameters_informations, verbose=True
        )
        self.transformation_method = transformation_method
        self.transformation_to_class_method = transformation_to_class_method
        self.optimizer = optimizer
        self.lambda_classification = lambda_classification
        self.lambda_segmentation = lambda_segmentation
        self.alpha = alpha
 
    def fit(
        self,
        signals,
        segmentations,
        true_classification,
        verbose=False,
        num_epochs=10,
        batch_size=5,
        test_batch_idx=[],
        static_params={},
        **fit_params
    ):
        """Implements a learning loop over epochs."""
        y = np.zeros(segmentations.shape)
        for i in range(len(segmentations)) :
            change_indx = find_change_indices(segmentations[i])
            ground_truth = np.zeros(segmentations[i].shape)
            ground_truth[change_indx[:-1]] = 1
            y[i] = ground_truth

        # Initialize static parameters
        for key, value in static_params.items():
            self.params[key] = value

        # Rest of the function remains the same
        train_loss = []
        acc_train = []
        acc_test = []
        # Initialize optimizer.

        opt_state = self.optimizer.init(self.params)

        # Loop over the training epochs

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loader = create_data_loader_classification(
                signals, segmentations,true_classification, batch_size, test_batch_idx
            )

            bacth_losses = []
            batch_acc_train = []
            batch_acc_test = []

            for batch_type, (data, target, true_classification_target) in train_loader:
                if batch_type == "Train":
                    data = jnp.array(data)
                    target = jnp.array(target)
                    true_classification_target = jnp.array(true_classification_target)
                    # Compute the loss and gradients using only dynamic parameters
                    value, grads = final_loss_and_grad_classification_bin(
                        self.params, self.transformation_method, self.transformation_to_class_method, data, target, true_classification_target, self.lambda_classification, self.lambda_segmentation, self.alpha
                    )
                    # Update only the dynamic parameters
                    updates, opt_state = self.optimizer.update(grads, opt_state)
                    self.params = optax.apply_updates(self.params, updates)

                    for key, value in static_params.items():
                        self.params[key] = value

                    bacth_losses.append(value)

                elif batch_type == "Test":
                    pass

            epoch_loss = np.mean(bacth_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time

            if verbose:
                print(
                    "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                        epoch + 1,
                        epoch_time,
                        epoch_acc_train,
                        epoch_acc_test,
                        epoch_loss,
                        jnp.exp(self.params["beta"]),
                    )
                )

        return train_loss, opt_state, self.params

    def predict(self, signals, height=0.5, distance = 50, threshold = 0., limit = 50.):
        predictions = []
        for i in range(len(signals)):
            transformed_signal = self.transformation_method(self.params, signals[i])
            peaks, _ = find_peaks(transformed_signal[:,0], height=height, distance = distance, threshold=threshold) 
            for element in peaks :
                if element < limit or element > transformed_signal.shape[0] - limit:
                    # Create a boolean mask to identify the element to delete
                    mask = peaks != element

                    # Use the mask to create a new array without the element to delete
                    peaks = peaks[mask]
    
            peaks = np.append(peaks, transformed_signal.shape[0])
            predictions.append(peaks)

        return predictions

    def display(self, signals, true_segmentations, true_classification):
        predictions = self.predict(signals)
        print("Used margin = ", true_segmentations.shape[1] * (5 / 100))
        print("--------------------------------------")
        for i in range(len(signals)):
            print("Signal n° ", i)
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signals[i])
            )
            precision, recall = precision_recall(
                np.array(find_change_indices(true_segmentations[i])),
                np.array(predictions[i]),
                margin=true_segmentations.shape[1] * (5 / 100),
            )
            
            print("True segmentation :", find_change_indices(true_segmentations[i]))
            print("Predicted segmentation :", predictions[i])
            if precision + recall == 0:
                print("F1 score :", 0)
            else:
                print("F1 score :", 2 * (precision * recall) / (precision + recall))
            rpt.display(
                np.array(transformed_signal),
                find_change_indices(true_segmentations[i]),
                predictions[i],
            )
            plt.show()
            # Plotting side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            predicted_label = self.transformation_to_class_method(self.params,transformed_signal)
            print("Classification loss : " ,compute_cross_entropy_loss(predicted_label,true_classification[i]))

            # Plot predicted labels
            for j in range(true_classification.shape[2]):  # assuming 6 classes
                axes[0].plot(predicted_label[:, j], label=f'Class {j}')

            axes[0].set_title('Predicted Labels')
            axes[0].legend()
            axes[0].grid(True)

            # Plot true labels
            for j in range(true_classification.shape[2]):  # assuming 6 classes
                axes[1].plot(true_classification[i][:, j], label=f'Class {j}')

            axes[1].set_title('True Labels')
            axes[1].legend()
            axes[1].grid(True)

            plt.show()