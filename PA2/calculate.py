from functools import reduce

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#konstanta za broj binova
NUM_BINS = 8

# Funkcija za obradu jednog piksela
def process_pixel(histograms, pixel):
    r, g, b = pixel
    bin_size = 256 // NUM_BINS

    # Izračunavamo indeks binova, i odmah ograničavamo ih na opseg 0 <= bin < NUM_BINS
    r_bin = r // bin_size
    g_bin = g // bin_size
    b_bin = b // bin_size

    # Ograničavamo binove da ne pređu granice (ako je vrednost veća od 255, ostaje u poslednjem binu)
    r_bin = min(r_bin, NUM_BINS - 1)
    g_bin = min(g_bin, NUM_BINS - 1)
    b_bin = min(b_bin, NUM_BINS - 1)

    histograms[0][r_bin] += 1
    histograms[1][g_bin] += 1
    histograms[2][b_bin] += 1
    return histograms

def calculate_normalized_bins_histograms(image_array):
    height = reduce(lambda acc, _: acc + 1, image_array[0], 0)
    width = reduce(lambda acc, _: acc + 1, image_array[1], 0)

    # Svi pikseli slike u jednoj listi
    all_pixels = image_array.reshape(-1, 3)

    # Inicijalizacija praznih histograma za R, G, i B
    initial_histograms = [
        np.zeros(NUM_BINS, dtype=np.float32),
        np.zeros(NUM_BINS, dtype=np.float32),
        np.zeros(NUM_BINS, dtype=np.float32)
    ]

    # Reduce za izračunavanje histograma
    r_hist, g_hist, b_hist = reduce(process_pixel, all_pixels, initial_histograms)

    # Normalizacija histograma
    total_pixels = height * width
    r_hist = reduce(lambda acc, x: acc + [x / total_pixels], r_hist, [])
    g_hist = reduce(lambda acc, x: acc + [x / total_pixels], g_hist, [])
    b_hist = reduce(lambda acc, x: acc + [x / total_pixels], b_hist, [])

    # Vraćamo rezultat kao numpy matricu
    return np.stack([r_hist, g_hist, b_hist], axis=0)

def cosine_similarity(hist1, hist2):
    # flattenovanje histogram matrica u 1D nizove
    flat_hist1 = hist1.flatten()
    flat_hist2 = hist2.flatten()

    dot_product = np.dot(flat_hist1, flat_hist2)  #skalarni proizvod histograma(vektora)
    norm1 = np.linalg.norm(flat_hist1)  #duzina prvog vektora
    norm2 = np.linalg.norm(flat_hist2)  #duzina drugog vektora

    #provera zbog deljenja s nulom
    if norm1 == 0 or norm2 == 0:
        #ako je jedan vektor nula, slicnost je 0
        return 0.0

    #kosinusna slicnost
    similarity = dot_product / (norm1 * norm2)
    return similarity

def plot_histograms(histograms, bins_num):
    colors = ['red', 'green', 'blue']  #boje za svaku komponentu
    labels = ['Red', 'Green', 'Blue']  #oznake za komponente

    #petlja kroz tri komponente(R, G, B)
    for i in range(3):
        plt.plot(histograms[i], color=colors[i], label=f'{labels[i]} Component')

    #plt.ylim(0, 1)  #Y-osa od 0 do 1
    plt.title('Normalized Color Histograms')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    print(f"BINS: ")
    bin_width = 256 // bins_num
    for i in range(bins_num):
        start = i * bin_width
        end = (i + 1) * bin_width
        print(f"\t Bin {i}: {start}-{end}")

    print("Normalized Histograms (RGB):")
    for i in range(3):
        print(f"\t {labels[i]} histogram:  {histograms[i]}")


def resize_image_to_32x32(image_path):
    """Učitaj sliku i konvertuj je u format (32, 32, 3)."""
    # Učitaj sliku sa datog puta
    image = Image.open(image_path)

    # Konvertuj sliku u RGB format (ako već nije)
    image = image.convert('RGB')

    # Resizuj sliku na 32x32 piksela
    image = image.resize((32, 32))

    # Konvertuj sliku u NumPy niz (sa oblikom (32, 32, 3))
    image_array = np.array(image)

    return image_array