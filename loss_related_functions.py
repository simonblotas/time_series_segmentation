import jax.numpy as jnp
from jax import jit
from jax import jit
from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
from typing import Dict, Callable, Tuple
from breakpoints_computation import get_optimal_projection, segmentation_to_projection


@jit
def compute_v_value(
    signal: jnp.ndarray, projection: jnp.ndarray, segmentation_size: int, beta: float
) -> float:
    """
    Computes the value of the function V for a given signal, projection, segmentation size, and penalty parameter.

    Parameters:
    signal (jnp.ndarray): The signal.
    projection (jnp.ndarray): The projection of the signal.
    segmentation_size (int): The size of the segmentation.
    beta (float): The penalty parameter.

    Returns:
    float: The computed value of the function V.
    """

    return ((signal - projection) ** 2).sum() + jnp.exp(beta) * segmentation_size


@jit
def loss(
    transformed_signal: jnp.ndarray, params: Dict, true_segmentation: jnp.ndarray
) -> float:
    """
    Computes the loss function for a given transformed signal, penalty
    parameter, and true segmentation.

    Parameters:
    transformed_signal (jnp.ndarray): The transformed signal.
    beta (float): The penalty parameter.
    true_segmentation (jnp.ndarray): The true segmentation points.

    Returns:
    float: The computed loss value.
    """
    # Calculate the projection and segment IDs using a prediction function
    pred_projection, pred_segmentation_size, segment_ids_pred = get_optimal_projection(
        transformed_signal, penalty=jnp.exp(params["beta"])
    )
    # Calculate the true projection and segmentation size
    true_projection = segmentation_to_projection(transformed_signal, true_segmentation)
    true_segmentation_size = true_segmentation[-1] + 1
    # Calculate the loss based on a difference in V values
    loss_value = (
        jnp.sum(
            compute_v_value(
                transformed_signal,
                true_projection,
                true_segmentation_size,
                params["beta"],
            )
            - compute_v_value(
                transformed_signal,
                pred_projection,
                pred_segmentation_size,
                params["beta"],
            )
        )
        / true_segmentation_size
    )
    return loss_value


def final_loss_and_grad(
    params: Dict,
    transformation: Callable,
    signals: jnp.ndarray,
    true_segmentation: jnp.ndarray,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Compute the final loss and gradients for a transformation applied to a list of signals.

    Args:
        params (Dict): The parameters for the transformation.
        transformation (Callable): A callable transformation function.
        signals (jnp.ndarray): The input signals as a numpy array of shape (nb_signals, signal_length).
        true_segmentation (jnp.ndarray): The true segmentation data as a numpy array.

    Returns:
        Tuple[float, Dict[str, jnp.ndarray]]: A tuple containing:
            - final_loss (float): The final loss computed.
            - grads (Dict[str, jnp.ndarray]): Gradients with respect to parameters as a dictionary.
    """

    def main_loss(
        params: Dict,
        transformation: Callable,
        signal: jnp.ndarray,
        true_segmentation: jnp.ndarray,
    ) -> float:
        transformed_signal = transformation(params, signal)
        return loss(transformed_signal, params, true_segmentation)

    batched_value_and_grad = vmap(
        value_and_grad(main_loss, argnums=0, allow_int=True),
        in_axes=(
            None,
            None,
            0,
            0,
        ),
        out_axes=0,
    )
    losses, grads = batched_value_and_grad(
        params, transformation, signals, true_segmentation
    )
    final_loss = jnp.sum(losses)
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)

    return final_loss, grads

def final_loss_and_grad_autoencoder(
    params: Dict,
    encoder: Callable,
    decoder: Callable,
    signals: jnp.ndarray,
    true_segmentation: jnp.ndarray,
    lambda_reconstruction: float, 
    lambda_segmentation: float, 
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def main_loss(
        params: Dict,
        encoder: Callable,
        decoder: Callable,
        signal: jnp.ndarray,
        true_segmentation: jnp.ndarray,
        lambda_reconstruction: float, 
        lambda_segmentation: float, 
    ) -> float:
        encoded_signal = encoder(params, signal)
        loss_segmentation = loss(encoded_signal, params, true_segmentation)
        decoded_signal = decoder(params, encoded_signal, true_segmentation)
        loss_reconstruction = jnp.sum(jnp.abs(signal-decoded_signal))
        return lambda_segmentation * loss_segmentation + lambda_reconstruction * loss_reconstruction

    batched_value_and_grad = vmap(
        value_and_grad(main_loss, argnums=0, allow_int=True),
        in_axes=(
            None,
            None,
            None,
            0,
            0,
            None,
            None,
        ),
        out_axes=0,
    )
    losses, grads = batched_value_and_grad(
        params, encoder, decoder, signals, true_segmentation, lambda_reconstruction, lambda_segmentation
    )
    final_loss = jnp.sum(losses)
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)

    return final_loss, grads

def BinaryCrossEntropy(y_true, y_pred, alpha = 1.):
    y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * jnp.log(1-y_pred + 1e-7)
    term_1 = alpha * y_true * jnp.log(y_pred + 1e-7)
    return -jnp.mean(term_0+term_1, axis=0)


def final_loss_and_grad_bin(
    params: Dict,
    transformation: Callable,
    signals: jnp.ndarray,
    true_segmentation: jnp.ndarray,
    alpha = 1,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Compute the final loss and gradients for a transformation applied to a list of signals.

    Args:
        params (Dict): The parameters for the transformation.
        transformation (Callable): A callable transformation function.
        signals (jnp.ndarray): The input signals as a numpy array of shape (nb_signals, signal_length).
        true_segmentation (jnp.ndarray): The true segmentation data as a numpy array.

    Returns:
        Tuple[float, Dict[str, jnp.ndarray]]: A tuple containing:
            - final_loss (float): The final loss computed.
            - grads (Dict[str, jnp.ndarray]): Gradients with respect to parameters as a dictionary.
    """
    
    def main_loss_bin(
        params: Dict,
        transformation: Callable,
        signal: jnp.ndarray,
        true_segmentation: jnp.ndarray,
        alpha = 1.,
    ) -> float:
        
        transformed_signal = transformation(params, signal)

        return BinaryCrossEntropy(transformed_signal[:,0],true_segmentation, alpha)


    batched_value_and_grad = vmap(
        value_and_grad(main_loss_bin, argnums=0),
        in_axes=(
            None,
            None,
            0,
            0,
            None,
        ),
        out_axes=0,
    )
    losses, grads = batched_value_and_grad(
        params, transformation, signals, true_segmentation, alpha
    )
    #print(grads)
    final_loss = jnp.sum(losses)
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)

    return final_loss, grads


def compute_cross_entropy_loss(predictions, true_labels):
    """
    Compute the cross-entropy loss for each element.

    Args:
    - predictions: jnp.array, shape (batch_size, num_classes) - Predicted probabilities for each class.
    - true_labels: jnp.array, shape (batch_size, num_classes) - True class labels with one-hot encoding.

    Returns:
    - jnp.array, shape (batch_size,) - Cross-entropy loss for each element.
    """
    # Clip the predicted probabilities to avoid numerical instability (log(0) is undefined)
    predictions = jnp.clip(predictions, 1e-15, 1.0 - 1e-15)

    # Compute the cross-entropy loss for each element
    loss_per_element = -jnp.sum(true_labels * jnp.log(predictions), axis=-1)

    return jnp.sum(loss_per_element)

def final_loss_and_grad_classification(
    params: Dict,
    transformation: Callable,
    transformation_to_class: Callable,
    signals: jnp.ndarray,
    true_segmentation: jnp.ndarray,
    true_classification: jnp.ndarray,
    lambda_classification =1.,
    lambda_segmentation =1.,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Compute the final loss and gradients for a transformation applied to a list of signals.

    Args:
        params (Dict): The parameters for the transformation.
        transformation (Callable): A callable transformation function.
        signals (jnp.ndarray): The input signals as a numpy array of shape (nb_signals, signal_length).
        true_segmentation (jnp.ndarray): The true segmentation data as a numpy array.

    Returns:
        Tuple[float, Dict[str, jnp.ndarray]]: A tuple containing:
            - final_loss (float): The final loss computed.
            - grads (Dict[str, jnp.ndarray]): Gradients with respect to parameters as a dictionary.
    """
    
    def main_loss_classification(
        params: Dict,
        transformation: Callable,
        transformation_to_class: Callable,
        signal: jnp.ndarray,
        true_segmentation: jnp.ndarray,
        true_classification: jnp.ndarray,
        lambda_classification =1.,
        lambda_segmentation =1.,
    ) -> float:
        
        transformed_signal = transformation(params, signal)
        class_pred_signal = transformation_to_class(params, transformed_signal)
        segmentation_loss = loss(transformed_signal, params, true_segmentation)
        classification_loss = compute_cross_entropy_loss(class_pred_signal, true_classification)
        return lambda_classification * classification_loss + segmentation_loss * lambda_segmentation


    batched_value_and_grad = vmap(
        value_and_grad(main_loss_classification, argnums=0),
        in_axes=(
            None,
            None,
            None,
            0,
            0,
            0,
            None,
            None,
        ),
        out_axes=0,
    )
    losses, grads = batched_value_and_grad(
        params, transformation, transformation_to_class, signals, true_segmentation, true_classification, lambda_classification, lambda_segmentation
    )
    final_loss = jnp.sum(losses)
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)

    return final_loss, grads

def final_loss_and_grad_classification_bin(
    params: Dict,
    transformation: Callable,
    transformation_to_class: Callable,
    signals: jnp.ndarray,
    true_segmentation: jnp.ndarray,
    true_classification: jnp.ndarray,
    lambda_classification =1.,
    lambda_segmentation =1.,
    alpha = 1,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Compute the final loss and gradients for a transformation applied to a list of signals.

    Args:
        params (Dict): The parameters for the transformation.
        transformation (Callable): A callable transformation function.
        signals (jnp.ndarray): The input signals as a numpy array of shape (nb_signals, signal_length).
        true_segmentation (jnp.ndarray): The true segmentation data as a numpy array.

    Returns:
        Tuple[float, Dict[str, jnp.ndarray]]: A tuple containing:
            - final_loss (float): The final loss computed.
            - grads (Dict[str, jnp.ndarray]): Gradients with respect to parameters as a dictionary.
    """
    
    def main_loss_classification_bin(
        params: Dict,
        transformation: Callable,
        transformation_to_class: Callable,
        signal: jnp.ndarray,
        true_segmentation: jnp.ndarray,
        true_classification: jnp.ndarray,
        lambda_classification =1.,
        lambda_segmentation =1.,
        alpha = 1,
    ) -> float:
        
        transformed_signal = transformation(params, signal)
        class_pred_signal = transformation_to_class(params, transformed_signal)
        segmentation_loss = BinaryCrossEntropy(transformed_signal[:,0],true_segmentation, alpha)
        classification_loss = compute_cross_entropy_loss(class_pred_signal, true_classification)
        return lambda_classification * classification_loss + segmentation_loss * lambda_segmentation


    batched_value_and_grad = vmap(
        value_and_grad(main_loss_classification_bin, argnums=0),
        in_axes=(
            None,
            None,
            None,
            0,
            0,
            0,
            None,
            None,
            None,
        ),
        out_axes=0,
    )
    losses, grads = batched_value_and_grad(
        params, transformation, transformation_to_class, signals, true_segmentation, true_classification, lambda_classification, lambda_segmentation, alpha
    )
    final_loss = jnp.sum(losses)
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)

    return final_loss, grads

