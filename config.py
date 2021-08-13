import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. 
# Choosing the learning rate is challenging as a value too small may result in a long training process that could get stuck, whereas a value too large may result 
# in learning a sub-optimal set of weights too fast or an unstable training process.
LEARNING_RATE = 1e-4

# While weight decay is an additional term in the weight update rule that causes the weights to exponentially decay to zero, if no other update is scheduled.
WEIGHT_DECAY = 5e-4

# The batch size defines the number of samples that will be propagated through the neural network (https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network)
BATCH_SIZE = 64

# The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
NUM_EPOCHS = 100

# tells the data loader instance how many sub-processes to use for data loading
NUM_WORKERS = 10

CHECKPOINT_FILE = "b3.pth.tar"
# not needed, useful if you have nvidia gpu
PIN_MEMORY = True

SAVE_MODEL = True
LOAD_MODEL = True

# data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=150, height=150),
        A.RandomCrop(height=120, width=120),
        A.Normalize(
            mean = [0.3199,0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        # ToTensor() is now deprecated and will be removed in future versions
        ToTensorV2()
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(width=120, height=120),
        A.Normalize(
            mean = [0.3199,0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
)