import slideflow as sf
from slideflow.mil import mil_config

# Create a MIL (Multiple Instance Learning) configuration object
config = mil_config(
    model='attention_mil',  # Use attention-based MIL architecture
    lr=1e-4,               # Learning rate set to 0.0001
    batch_size=16,         # Process 16 samples per batch
    epochs=20,             # Train for 20 complete passes through the dataset
    bag_size=1024,         # Number of instances in each bag (group of instances)
    fit_one_cycle=True     # Use one-cycle learning rate scheduling for better convergence
)