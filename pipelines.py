from models import SimplePipeline, PipelineAutoencoder, SimplePipelineBin, SimplePipelineClassification, SimplePipelineClassificationBin
from network_layers_definitions import  initialize_network, convolution_layer, dense_layer, normalize_signal, transposed_convolution_layer
import jax
import jax.numpy as jnp
from flax import linen as nn


##------------------------- Pipeline n°1 | Transformation + LOSS Segmentation ----------------------

@jax.jit
def transformation_method(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_conv_1 = convolution_layer(params['conv_layer_1_filter_weights'], params['conv_layer_1_bias'], input_, stride = 4 )
    out_relu_1 = nn.relu(out_conv_1)
    out_conv_2 = convolution_layer(params['conv_layer_2_filter_weights'], params['conv_layer_2_bias'], out_relu_1, stride = 4 )
    out_relu_2 = nn.relu(out_conv_2)
    output = normalize_signal(out_relu_2[0])
    return output 

layer_sizes = []
conv_layer_params = [(1, 300, 3, 3), (1, 300, 3, 2)]
tr_conv_layer_params = []
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]

transformation_pipeline = SimplePipeline(transformation_method, initialize_network, parameters_informations)

transformed_signal_length = 1 / 4

##------------------------- Pipeline n°2 | Transformation + LOSS BINARY classification ----------------------

@jax.jit
def transformation_method_bin(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_conv_1 = convolution_layer(params['conv_layer_1_filter_weights'], params['conv_layer_1_bias'], input_, stride = 4 )
    out_relu_1 = nn.relu(out_conv_1)
    out_conv_2 = convolution_layer(params['conv_layer_2_filter_weights'], params['conv_layer_2_bias'], out_relu_1, stride = 4 )
    out_relu_2 = nn.relu(out_conv_2)
    output = normalize_signal(out_relu_2[0])
    softmax_result = jnp.exp(output) / jnp.sum(jnp.exp(output), axis=1, keepdims=True)
    return softmax_result

layer_sizes = []
conv_layer_params = [(1, 300, 3, 3), (1, 300, 3, 2)]
tr_conv_layer_params = []
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]

transformation_pipeline_bin = SimplePipelineBin(transformation_method_bin, initialize_network, parameters_informations, alpha=1000)

transformed_signal_length_bin = 1 / 4

##------------------------- Pipeline n°3 | Fourier + Transformation + LOSS Segmentation ----------------------


@jax.jit
def transformation_method_fourier(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_conv_1 = convolution_layer(params['conv_layer_1_filter_weights'], params['conv_layer_1_bias'], input_)
    out_relu_1 = nn.relu(out_conv_1)
    out_conv_2 = convolution_layer(params['conv_layer_2_filter_weights'], params['conv_layer_2_bias'], out_relu_1)
    out_relu_2 = nn.relu(out_conv_2)
    out_dense_1 = dense_layer(params['linear_layer_1_weights'], params['linear_layer_1_bias'], out_relu_2)
    out_dense_2 = dense_layer(params['linear_layer_2_weights'], params['linear_layer_2_bias'], out_dense_1)
    output = normalize_signal(out_dense_2[0])
    return output 

nperseg=300
n_dims = int(nperseg/2 +1)
noverlap=292
# Example usage with linear and convolutional layers
layer_sizes = [(10,n_dims), (2,10)]
conv_layer_params = [(1, 300, n_dims, n_dims), (1, 300, n_dims, n_dims)]  # (Broadcast dimension, temporal_lenght, input_dimsension, output_dimension)
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]

pipeline_fourier = SimplePipeline(transformation_method_fourier, initialize_network, parameters_informations)

transformed_signal_length_fourier = 1 / (nperseg - noverlap)

##------------------------- Pipeline n°4 | Fourier + Transformation + LOSS BINARY classification ----------------------

@jax.jit
def transformation_method_fourier_bin(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_conv_1 = convolution_layer(params['conv_layer_1_filter_weights'], params['conv_layer_1_bias'], input_)
    out_relu_1 = nn.relu(out_conv_1)
    out_conv_2 = convolution_layer(params['conv_layer_2_filter_weights'], params['conv_layer_2_bias'], out_relu_1)
    out_relu_2 = nn.relu(out_conv_2)
    out_dense_1 = dense_layer(params['linear_layer_1_weights'], params['linear_layer_1_bias'], out_relu_2)
    out_dense_2 = dense_layer(params['linear_layer_2_weights'], params['linear_layer_2_bias'], out_dense_1)
    output = normalize_signal(out_dense_2[0])
    softmax_result = jnp.exp(output) / jnp.sum(jnp.exp(output), axis=1, keepdims=True)
    return softmax_result

nperseg=300
n_dims = int(nperseg/2 +1)
noverlap=292
# Example usage with linear and convolutional layers
layer_sizes = [(10,n_dims), (2,10)]
conv_layer_params = [(1, 300, n_dims, n_dims), (1, 300, n_dims, n_dims)]  # (Broadcast dimension, temporal_lenght, input_dimsension, output_dimension)
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]

pipeline_fourier_bin = SimplePipeline(transformation_method_fourier_bin, initialize_network, parameters_informations)

transformed_signal_length_fourier_bin = 1 / (nperseg - noverlap)


##------------------------- Pipeline n°5 |Transformation + SEG + Loss Classification ----------------------

@jax.jit
def transformation_to_class_method(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_dense_1 = dense_layer(params['linear_layer_1_weights'], params['linear_layer_1_bias'], input_)
    out_dense_2 = dense_layer(params['linear_layer_2_weights'], params['linear_layer_2_bias'], out_dense_1)
    #output = normalize_signal(out_dense_1[0])
    output = out_dense_2[0]
    softmax_result = jnp.exp(output) / jnp.sum(jnp.exp(output), axis=1, keepdims=True)
    return softmax_result


lambda_segmentation = 1.
lambda_classification = 1.
conv_layer_params = [(1, 100, 3, 3), (1, 100, 3, 2)]
tr_conv_layer_params = []
transformed_signal_length_seg_class = 1 / 4

# ---- 6 classes ---- :
nb_class = 6
layer_sizes = [(10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_seg_class_6 = SimplePipelineClassification(transformation_method, transformation_to_class_method, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification)

# ---- 7 classes ---- :
nb_class = 7
layer_sizes = [(10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_seg_class_7 = SimplePipelineClassification(transformation_method, transformation_to_class_method, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification)

# ---- 13 classes ---- :
nb_class = 13
layer_sizes = [(10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_seg_class_13 = SimplePipelineClassification(transformation_method, transformation_to_class_method, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification)


##------------------------- Pipeline n°6- | FOURIER + SEG + Loss Classification ----------------------

@jax.jit
def transformation_to_class_method_fourier(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_dense_1 = dense_layer(params['linear_layer_3_weights'], params['linear_layer_3_bias'], input_)
    out_dense_2 = dense_layer(params['linear_layer_4_weights'], params['linear_layer_4_bias'], out_dense_1)
    #output = normalize_signal(out_dense_1[0])
    output = out_dense_2[0]
    softmax_result = jnp.exp(output) / jnp.sum(jnp.exp(output), axis=1, keepdims=True)
    return softmax_result


lambda_segmentation = 1.
lambda_classification = 1.
nperseg=300
n_dims = int(nperseg/2 +1)
noverlap=292
# Example usage with linear and convolutional layers
conv_layer_params = [(1, 300, n_dims, n_dims), (1, 300, n_dims, n_dims)]  # (Broadcast dimension, temporal_lenght, input_dimsension, output_dimension)
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]

transformed_signal_length_fourier_seg_class = 1 / (nperseg - noverlap)


# ---- 6 classes ---- :
nb_class = 6
layer_sizes = [(10,n_dims), (2,10), (10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_fourier_seg_class_6 = SimplePipelineClassification(transformation_method_fourier, transformation_to_class_method_fourier, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification)

# ---- 7 classes ---- :
nb_class = 7
layer_sizes = [(10,n_dims), (2,10), (10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_fourier_seg_class_7 = SimplePipelineClassification(transformation_method_fourier, transformation_to_class_method_fourier, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification)

# ---- 13 classes ---- :
nb_class = 13
layer_sizes = [(10,n_dims), (2,10), (10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_fourier_seg_class_13 = SimplePipelineClassification(transformation_method_fourier, transformation_to_class_method_fourier, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification)

##------------------------- Pipeline n°7- | Transformation + BIN + Loss Classification ----------------------

@jax.jit
def transformation_to_class_method_bin(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_dense_1 = dense_layer(params['linear_layer_1_weights'], params['linear_layer_1_bias'], input_)
    out_dense_2 = dense_layer(params['linear_layer_2_weights'], params['linear_layer_2_bias'], out_dense_1)
    #output = normalize_signal(out_dense_1[0])
    output = out_dense_2[0]
    softmax_result = jnp.exp(output) / jnp.sum(jnp.exp(output), axis=1, keepdims=True)
    return softmax_result

alpha = 1000
lambda_segmentation = 1.
lambda_classification = 1.
conv_layer_params = [(1, 100, 3, 3), (1, 100, 3, 2)]
tr_conv_layer_params = []
transformed_signal_length_bin_class = 1 / 4

# ---- 6 classes ---- :
nb_class = 6
layer_sizes = [(10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_bin_class_6 = SimplePipelineClassificationBin(transformation_method_bin, transformation_to_class_method_bin, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification, alpha = alpha)

# ---- 7 classes ---- :
nb_class = 7
layer_sizes = [(10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_bin_class_7 = SimplePipelineClassificationBin(transformation_method_bin, transformation_to_class_method_bin, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification,alpha = alpha)

# ---- 13 classes ---- :
nb_class = 13
layer_sizes = [(10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_bin_class_13 = SimplePipelineClassificationBin(transformation_method_bin, transformation_to_class_method_bin, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification,alpha = alpha)

##------------------------- Pipeline n°8- | FOURIER + BIN + Loss Classification ----------------------

@jax.jit
def transformation_to_class_method_fourier_bin(params, in_array):
    """ Compute the forward pass for each example individually """
    input_ = jnp.expand_dims( in_array, axis = 0)
    out_dense_1 = dense_layer(params['linear_layer_3_weights'], params['linear_layer_3_bias'], input_)
    out_dense_2 = dense_layer(params['linear_layer_4_weights'], params['linear_layer_4_bias'], out_dense_1)
    #output = normalize_signal(out_dense_1[0])
    output = out_dense_2[0]
    softmax_result = jnp.exp(output) / jnp.sum(jnp.exp(output), axis=1, keepdims=True)
    return softmax_result

alpha = 1000
lambda_segmentation = 1.
lambda_classification = 1.
nperseg=300
n_dims = int(nperseg/2 +1)
noverlap=292
# Example usage with linear and convolutional layers
conv_layer_params = [(1, 300, n_dims, n_dims), (1, 300, n_dims, n_dims)]  # (Broadcast dimension, temporal_lenght, input_dimsension, output_dimension)
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]

transformed_signal_length_fourier_bin_class = 1 / (nperseg - noverlap)


# ---- 6 classes ---- :
nb_class = 6
layer_sizes = [(10,n_dims), (2,10), (10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_fourier_bin_class_6 = SimplePipelineClassificationBin(transformation_method_fourier_bin, transformation_to_class_method_fourier_bin, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification, alpha = alpha)

# ---- 7 classes ---- :
nb_class = 7
layer_sizes = [(10,n_dims), (2,10), (10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_fourier_bin_class_7 = SimplePipelineClassificationBin(transformation_method_fourier_bin, transformation_to_class_method_fourier_bin, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification, alpha = alpha)

# ---- 13 classes ---- :
nb_class = 13
layer_sizes = [(10,n_dims), (2,10), (10,2),(nb_class,10)]
parameters_informations = [layer_sizes, conv_layer_params, tr_conv_layer_params]
transformation_pipeline_fourier_bin_class_13 = SimplePipelineClassificationBin(transformation_method_fourier_bin, transformation_to_class_method_fourier_bin, initialize_network, parameters_informations, lambda_segmentation=lambda_segmentation, lambda_classification=lambda_classification, alpha = alpha)




# Creation of the dict :

pipelines = {
             "trans_seg": { 'pipeline':transformation_pipeline,'transformed_signal_length' : transformed_signal_length, 'fourier' : False, 'class' : False},
             "trans_bin": {'pipeline':transformation_pipeline_bin,'transformed_signal_length' : transformed_signal_length_bin, 'fourier' : False, 'class' : False },
             "trans_seg_fourier": { 'pipeline':pipeline_fourier,'transformed_signal_length' : transformed_signal_length_fourier, 'fourier' : True, 'class' : False},
             "trans_bin_fourier": { 'pipeline':pipeline_fourier_bin,'transformed_signal_length' : transformed_signal_length_fourier_bin, 'fourier' : True, 'class' : False},
             "trans_seg_class": { 'pipeline':{6 : transformation_pipeline_seg_class_6, 7 : transformation_pipeline_seg_class_7, 13 : transformation_pipeline_seg_class_13},'transformed_signal_length' : transformed_signal_length_seg_class, 'fourier' : False, 'class' : True},
             "trans_bin_class": { 'pipeline':{6 : transformation_pipeline_bin_class_6, 7 : transformation_pipeline_bin_class_7, 13 : transformation_pipeline_bin_class_13},'transformed_signal_length' : transformed_signal_length_bin_class, 'fourier' : False, 'class' : True},
             "trans_seg_class_fourier": { 'pipeline':{6 : transformation_pipeline_fourier_seg_class_6, 7 : transformation_pipeline_fourier_seg_class_7, 13 : transformation_pipeline_fourier_seg_class_13},'transformed_signal_length' : transformed_signal_length_fourier_seg_class, 'fourier' : True, 'class' : True},
             "trans_bin_class_fourier": { 'pipeline':{6 : transformation_pipeline_fourier_bin_class_6, 7 : transformation_pipeline_fourier_bin_class_7, 13 : transformation_pipeline_fourier_bin_class_13},'transformed_signal_length' : transformed_signal_length_fourier_bin_class, 'fourier' : True, 'class' : True},
             }