### What is data leakage?
Data leakage occurs when information from outside the training dataset is used to train the model. When doing dataset splitting, data leakage happens if the same sample appears in both the training or validation/test sets. This may increase the evaluation performance, since the model has “seen” part of the test data during training. However, the results may not be as good as it shows in reality.
### How do I avoid data leakage?
In my implementation of `dataset_loader.py`, I followed these principles:

1. Immediate Splitting: The dataset is divided into train, validation, and test sets as soon as the images are loaded, before any normalization or preprocessing.
2. Checking overlapping between Sets: I write a function that checks no image path appears in more than one split. This ensures complete separation between training, validation, and test sets.
3. Splitting from class independently: Each class (0–9) is split separately with the same ratio (70%/15%/15%).