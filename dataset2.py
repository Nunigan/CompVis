
def _parse_function(image_filename, label_filename, channels: int):
    """
    Parse image and label and return them. The image is divided by 255.0 and returned as float,
    the label is returned as is in uint8 format.
    Args:
        image_filename: name of the image file
        label_filename: name of the label file
        channels: channels of the input image, (the label is always one channel)
    Returns:
        tensors for the image and label read operations
    """
    image_string = tf.io.read_file(image_filename)
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

    # normalize image to zero mean
    image = tf.multiply(image_decoded, 2.0)
    image = tf.subtract(image, 1.0)

    label_string = tf.io.read_file(label_filename)
    label = tf.image.decode_png(label_string, dtype=tf.uint8, channels=1)

    return image, label


def create_dataset(data_dir: str, nr_classes: int,
                   patch_size: int, border_size:int, channels:int):
    path = Path(data_dir)
    image_files = list(path.glob('image*.png'))
    label_files = list(path.glob('label*.png'))

    # make sure they are in the same order
    image_files.sort()
    label_files.sort()

    image_files_array = np.asarray([str(p) for p in image_files])
    label_files_array = np.asarray([str(p) for p in label_files])

    dataset = tf.data.Dataset.from_tensor_slices((image_files_array,
                                                  label_files_array))
    # shuffle the filename, unfortunately, then we cannot cache them
    dataset = dataset.shuffle(buffer_size=10000)
    # read the images
    dataset = dataset.map(lambda image, file:
                          _parse_function(image, file, channels))
    # Set the sizes of the input image, as keras needs to know them
    dataset = dataset.map(lambda x, y:
                          (tf.reshape(x, shape=(patch_size, patch_size, channels)), y))
    # cut center of the label image in order to use valid filtering in the network
    b = border_size
    if b != 0:
        dataset = dataset.map(lambda x, y:
                              (x, y[b:-b, b:-b, :]))
    # no one hot encoding for sparse catecorical
    return dataset, image_files_array.size

