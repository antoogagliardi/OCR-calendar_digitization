import matplotlib.pyplot as plt
import numpy as np
import os
import collections

"""
Compute the class frequencies for a dataset based on the labels.
    Parameters:
        - dataset (list): A list of dictionary entries, where each entry represents a sample
            from the dataset. Each entry have a "label" key containing the label information.
    Returns:
        - total_samples (int): Total number of samples in the dataset.
        - frequencies (dict): A dictionary containing the absolute frequencies of each unique
            label character in the dataset.
        - scaled_frequencies (dict): A dictionary containing the relative frequencies
            (scaled by total_samples) of each unique label character in the dataset.
"""
def compute_class_frequency(dataset):
    total_samples = len(dataset)

    frequencies = collections.Counter()
    for entry in dataset:
        label = list(entry["label"])
        for char in label:
            frequencies[char] += 1 
    frequencies = dict(sorted(frequencies.items()))             # Sort dictionary keys
  
    scaled_frequencies = {}
    for key, val in frequencies.items():
        scaled_frequencies[key] = val/total_samples

    return total_samples, frequencies, scaled_frequencies

"""
Plot a bar chart illustrating the class frequencies based on a provided frequency dictionary.
    Parameters:
        - frequencies (dict): A dictionary containing the class frequencies, where keys represent
            class names or labels, and values represent their respective frequencies.
    Returns:
        - None: This function generates and displays a bar chart but does not return any value.
"""
def plot_class_frequency(frequencies):    
    classes = list(frequencies.keys())
    freq = list(frequencies.values())
    
    fig, ax = plt.subplots()                                                # Set up the figure and axes
    fig.set_size_inches(8, 5)
    bar_width = 0.5                                                         # Set the bar width

    x_pos = range(len(classes))                                             # Set the positions of the bars on the x-axis
    ax.bar(x_pos, freq, bar_width, align='center', color='lightskyblue')    # Plot the histogram
    
    # Set the x-axis labels to the class names
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)

    # Set the y-axis label
    ax.set_ylabel('Frequency')
    scale_factor = 5
    plt.yticks(np.arange(1, max(freq), int(max(freq)/scale_factor)))

    # Set the title of the plot
    ax.set_title('Class Frequency')

    # Increase space between xticks
    plt.tight_layout()

    # Display the plot
    plt.show()

"""
Plot a confusion matrix to visualize the performance of a classification model.
    Parameters:
        - cm (numpy.ndarray): Confusion matrix, where rows represent true labels and columns
            represent predicted labels.
        - out_class (list): List of class labels corresponding to the classes in the dataset.
        - cmap (str): Colormap for the plot (default is "Reds").
        - dim (tuple): Dimensions of the plot (height, width).
        - title (str): Title for the plot.
        - save_cm (bool): If True, save the confusion matrix plot as a PNG image.
        - save_dir (str): Directory to save the confusion matrix plot if save_cm is True.
        - file_name (str): File name for the saved confusion matrix plot.
        - verbose (bool): If True, display the plot interactively.
    Returns:
        - None: This function generates and optionally saves or displays a confusion matrix plot.
"""
def plot_confusion_matrix(cm, out_class:list, cmap:str="Reds", dim:tuple=(10,20), title:str=None, save_cm:bool=False, save_dir:str=None, file_name:str=None,
                          verbose:bool=False):
    # Use the labels that are in our dataset
    classes = out_class
    
    # if normalize:
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm[np.isnan(cm)] = 0                                            # Handle NaNs (zero division)

    fig, ax = plt.subplots()
    fig.set_size_inches(h=dim[0], w=dim[1])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Main Loop over data dimensions
    fmt = '.2f' # if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]*100, fmt) + "%",
                    ha="center", va="center",
                    color="black")                                  # Set text color to black

    fig.tight_layout()
    # Save the plot as a PNG image
    if save_cm == True:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    if verbose: plt.show(ax)