
def get_plot_file_name(model_type, plot_type, time_stamp, image_size):
    """
    plot_type: 'acc', 'loss'
    model_type: 'cnn', 'nn', etc.
    image_size: (width, height)
    Returns the name of the plot file.
    """
    return f'./plots/{model_type}/{time_stamp}_{plot_type}_size_{image_size[0]}_{image_size[1]}.jpeg'