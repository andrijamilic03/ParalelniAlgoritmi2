from functools import reduce
from itertools import takewhile

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from PA2.calculate import *
from cifar10 import load_balanced_cifar10
from functools import reduce

if __name__ == '__main__':

    balanced_cifar = load_balanced_cifar10(samples_per_class=100)

    # Definišite koje klase želite da uključite
    selected_classes = ['airplane', 'automobile', 'cat', 'dog', 'ship']

    # Filtrirajte klase koristeći funkcionalni pristup
    filtered_cifar = dict(filter(lambda item: item[0] in selected_classes, balanced_cifar.items()))

    # Izračunaj prosečne histograme po klasama
    '''average_histograms = dict(
        map(lambda class_entry: (class_entry[0], np.mean(np.array(list(map(calculate_normalized_bins_histograms, class_entry[1]))), axis=0)), filtered_cifar.items())
    )'''

    average_histograms = dict(
        map(
            lambda class_entry: (
                class_entry[0],
                reduce(
                    lambda acc, elem: (acc[0] + elem, acc[1] + 1),
                    map(calculate_normalized_bins_histograms, class_entry[1]),
                    (np.zeros_like(calculate_normalized_bins_histograms(class_entry[1][0])), 0)
                )[0] / reduce(
                    lambda acc, _: acc + 1,
                    class_entry[1], 0
                )
            ),
            filtered_cifar.items()
        )
    )

    image_path1 = 'content/auto.png'
    image_array1 = resize_image_to_32x32(image_path1)
    hist1 = calculate_normalized_bins_histograms(image_array1)

    similarities = map(lambda entry: (entry[0], cosine_similarity(entry[1], hist1)), average_histograms.items())

    sorted_classes = reduce(lambda acc, x: x if x[1] > acc[1] else acc, similarities, ("", -1))

    # Pronađi klasu sa najvećom sličnosti
    best_match_class, best_similarity = sorted_classes
    print(f"\nNajbliža klasa: {best_match_class} sa sličnosti: {best_similarity:.4f}")

