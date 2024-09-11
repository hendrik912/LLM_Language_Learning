import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ------------------------------------------------------

def plot_word_frequencies(sorted_word_tuples, threshold=0.25, figsize=(6,3)):
    
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    plt.rcParams['font.family'] = 'Nirmala UI'

    all_freqs = np.array([float(freq[1]) for freq in sorted_word_tuples])
    all_freqs /= np.sum(all_freqs)

    total_proportion = 0
    n = 0

    while total_proportion < threshold:
        total_proportion += all_freqs[n] 
        n += 1

    print(f"Threshold: {threshold}, number of words: {n}")

    top_n_words = [word[0] for word in sorted_word_tuples[:n]]
    top_n_freqs = [freq[1] for freq in sorted_word_tuples[:n]]

    plt.figure(figsize=figsize)
    plt.bar(top_n_words, top_n_freqs, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.ylim(0, top_n_freqs[0]) 

    plt.title(f'Top {n} Words and Their Frequencies')
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------

def plot_cumulative_coverage(sorted_word_tuples, limit=1000):
    
    frequency_values = [f for _, f in sorted_word_tuples]
    
    if limit is not None:
        frequency_values = frequency_values[:limit]

    cumulative_sum = [sum(frequency_values[:i+1]) for i in range(len(frequency_values))]

    total_sum = sum(frequency_values)

    cumulative_percentage = [(sum_values / total_sum) * 100 for sum_values in cumulative_sum]

    # Plotting
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(frequency_values) + 1), cumulative_percentage, marker='o') # , linestyle='-')

    plt.xlabel('Number of Words')
    plt.ylabel('Cumulative Percentage of Total Frequency')
    plt.title('Cumulative Distribution of Word Frequencies')
    plt.grid(True)
    # plt.xticks(range(1, len(frequency_values) + 1))
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# ------------------------------------------------------

