import jax
import jax.lax as lax
from jax import random
import jax.numpy as jnp

@jax.jit
def normalize_signal(signal):
    min_vals = jnp.min(signal, axis=0)
    max_vals = jnp.max(signal, axis=0)
    range_vals = max_vals - min_vals
    range_vals = jnp.where(range_vals < 1e-10, 1.0, range_vals)
    out = (signal - min_vals) / range_vals
    return out


# Function to initialize parameters with Gaussian (normal) distribution
def normal_initializer(shape, key, scale=1):
    return scale * jax.random.normal(key, shape)

def convolution_layer(kernel, bias, x, padding = "SAME", stride=2, lhs_dilation = (1,), rhs_dilation = (1,)):
    """Simple convolutionnal layer"""
    dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ("NWC", "WIO", "NWC"))

    out_conv = lax.conv_general_dilated(
        x, kernel, window_strides = (stride // 2,), padding = padding, lhs_dilation = lhs_dilation, rhs_dilation = rhs_dilation,  dimension_numbers = dn
    )


    return out_conv

def transposed_convolution_layer(kernel, bias, x, stride=2, rhs_dilation = (1,)):
    """Simple transposed convolutional layer"""
    dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ("NWC", "WIO", "NWC"))

    out_conv_transpose = lax.conv_transpose(x, kernel, strides = (stride // 2,), padding = "SAME", dimension_numbers = dn, rhs_dilation = rhs_dilation)

    return out_conv_transpose


def dense_layer(weights, bias, x):
    """Simple dense layer for single sample"""
    return jnp.dot(x, weights)


def initialize_linear_layer(m, n, key, scale=1):
    """Initialize weights for a linear (fully connected) layer"""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))



def initialize(sizes, key):
    """Initialize the weights of all layers of a linear layer network"""
    keys = random.split(key, len(sizes))
    return [
        initialize_linear_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def initialize_network(
    parameters_informations, beta_initial=jnp.log(10.0), verbose=False
):
    linear_layer_sizes, conv_layer_sizes, tr_conv_layer_sizes = (
        parameters_informations[0],
        parameters_informations[1],
        parameters_informations[2],
    )
    key = random.PRNGKey(0)
    if len(linear_layer_sizes) >0 :
        linear_params = [
            initialize(linear_layer, key) for linear_layer in linear_layer_sizes
        ]
    else:
        linear_params = []

    if len(conv_layer_sizes) > 0:
        conv_params = [
            normal_initializer(kernel_size, key) for kernel_size in conv_layer_sizes
        ]
    else:
        conv_params = []

    if len(tr_conv_layer_sizes) > 0:
        tr_conv_params = [
            normal_initializer(kernel_size, key) for kernel_size in tr_conv_layer_sizes
        ]
    else:
        tr_conv_params = []

    # Create a dictionary 'params' to store all the parameters
    params = {}

    # Store linear parameters in the dictionary
    for i, layer_param in enumerate(linear_params):
        params[f"linear_layer_{i+1}_weights"] = layer_param[0][0]
        params[f"linear_layer_{i+1}_bias"] = layer_param[0][1]

    # Store convolutional parameters in the dictionary
    for i, layer_param in enumerate(conv_params):
        params[f"conv_layer_{i+1}_filter_weights"] = layer_param[0]
        params[f"conv_layer_{i+1}_bias"] = layer_param[1]

    # Store convolutional parameters in the dictionary
    for i, layer_param in enumerate(tr_conv_params):
        params[f"tr_conv_layer_{i+1}_filter_weights"] = layer_param[0]
        params[f"tr_conv_layer_{i+1}_bias"] = layer_param[1]

    # Store Beta parameter in the dictionary
    params[f"beta"] = jnp.array(beta_initial)

    if verbose:
        # Print the parameters in the 'params' dictionary
        print("Parameters:")
        for name, value in params.items():
            print(f"{name} - Shape: {value.shape}")

    return params
