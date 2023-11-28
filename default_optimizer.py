import optax

start_learning_rate = 1e-2
# Exponential decay of the learning rate.
scheduler = optax.exponential_decay(
    init_value=start_learning_rate, transition_steps=1000, decay_rate=0.99
)

# Combining gradient transforms using `optax.chain`.
gradient_transform = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
    optax.scale_by_adam(),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0),
)