import os
import random

def load_dataset(file_root, seed):
    '''
    return dictionary: {label: file_name}
    '''
    dataset = {}
    for folder in sorted(os.listdir(file_root)):
        folder_path = os.path.join(file_root, folder)
        if not os.path.isdir(folder_path): continue # ensure only folders are processed
        label = folder_path[-1]
        # print(label) output: 0 1 2 ... 9
        files = []
        for file in sorted(os.listdir(folder_path)):
            if not file.endswith(".png"): continue
            files.append(os.path.join(folder, file))
        random.seed(seed)
        random.shuffle(files)
        dataset[label] = files
    return dataset

def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    '''
    based on given ratio, extract files from shuffled dataset
    '''
    train, val, test = [], [], []
    for label, files in dataset.items():
        N = len(files)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        n_test = N - n_train - n_val
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        train.extend([(f, label) for f in train_files])
        val.extend([(f, label) for f in val_files])
        test.extend([(f, label) for f in test_files])
    return train, val, test

def save_list(pairs, filename):
    with open(filename, "w") as f:
        for path, label in pairs:
            f.write(f"{path} {label}\n")

if __name__ == "__main__":

    # === variables ===
    file_root = "./MNIST"
    seed = 42
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    # === variables ===

    dataset = load_dataset(file_root, seed)
    train, val, test = split_dataset(dataset, train_ratio, val_ratio, test_ratio)

    save_list(train, "train_list.txt")
    save_list(val, "val_list.txt")
    save_list(test, "test_list.txt")
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")