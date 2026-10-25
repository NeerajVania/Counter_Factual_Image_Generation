from collections import defaultdict
import numpy as np

def make_balanced_val_split(dataset, n_val):
    labels = dataset.get_labels()          # list of ints
    labels = np.array(labels)

    classes = np.unique(labels)
    n_classes = len(classes)
    assert n_val % n_classes == 0, f"n_val must be divisible by number of classes ({n_classes})"

    per_class = n_val // n_classes

    val_indices = []
    train_indices = []

    for c in classes:
        class_idxs = np.where(labels == c)[0]
        np.random.shuffle(class_idxs)

        val_indices.extend(class_idxs[:per_class])
        train_indices.extend(class_idxs[per_class:])

    return train_indices, val_indices
