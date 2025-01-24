import numpy as np

def print_distribution(dataset, dataset_name, total_images=None, spacing=20):
    total_images = len(dataset) if total_images is None else total_images
    print(f"\n--- {dataset_name} Distribution ---")
    unique, counts = np.unique(dataset, return_counts=True)
    print(f"{'Morphology':^{spacing}}|{'Count':^{spacing}}|{'Percentage':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    for i in range(len(unique)):
        print(f"{unique[i]:^{spacing}}|{counts[i]:^{spacing}}|{f'{counts[i]/len(dataset)*100:.2f}%':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    print(f"Total Images: {len(dataset)}/{total_images}\t{len(dataset)/total_images*100:.2f}%\n")


def print_train_val_distribution(X_train, y_train, X_val, y_val, spacing=15, name='Train and Validation Data Morphology Distribution'):
    # Train and Validation Final Morphology Distribution
    print(f"{'-'*spacing*5}----")
    print(f"{name:^{spacing*5}}")
    print(f"{'-'*spacing*5}----")

    unique, counts_train = np.unique(y_train, return_counts=True)
    unique, counts_val = np.unique(y_val, return_counts=True)

    print(f"{'Morphology':^{spacing}}|{'Count_Train':^{spacing}}|{'Perc_of_Train':^{spacing}}|", end='')
    print(f"{'Count_Val':^{spacing}}|{'Perc_of_Val':^{spacing}}")
    print(f"{'-'*spacing*5}----")

    for i in range(len(unique)):
        print(f"{unique[i]:^{spacing}}|{counts_train[i]:^{spacing}}|{f'{counts_train[i]/len(y_train)*100:.2f}%':^{spacing}}|", end='')
        print(f"{counts_val[i]:^{spacing}}|{f'{counts_val[i]/len(y_val)*100:.2f}%':^{spacing}}")

    print(f"{'-'*spacing*5}----")
    print(f"{'Totals':^{spacing}}|{len(y_train):^{spacing}}|{'100.00%':^{spacing}}|", end='')
    print(f"{len(y_val):^{spacing}}|{'100.00%':^{spacing}}")

    print(f"{'-'*spacing*5}----")
    print(f"{f'TOTAL TRAINVAL IMAGES: {len(X_train)+len(X_val)}':^{spacing*5}}")
    print(f"{'-'*spacing*5}----")