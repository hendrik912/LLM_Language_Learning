
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import nltk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from IPython.display import display, clear_output
import ipywidgets as widgets
from IPython.display import display
from lang import utils

# ------------------------------------------------------

def plot_frequencies(sorted_word_tuples, threshold=0.5, figsize=(6, 3), highlight_words=None, hide_xticks=False, animate=True, logscale=True, title=""):
   
    # Path to the custom font in your project folder
    font_path = '/share/documents/students/heilts/Hiwi/lang/data/NotoSansDevanagari.ttf'
    prop = fm.FontProperties(fname=font_path)

    if highlight_words is None:
        highlight_words = []

    # Normalize frequencies
    all_freqs = np.array([float(freq[1]) for freq in sorted_word_tuples])
    all_freqs /= np.sum(all_freqs)

    # Determine the number of words needed to reach the threshold
    total_proportion = 0
    n = 0
    while total_proportion < threshold:
        total_proportion += all_freqs[n]
        n += 1

    if not animate:
        print(f"Threshold: {threshold}, number of words: {n}")

    # Extract top n words and their frequencies
    top_n_words = [word[0] for word in sorted_word_tuples[:n]]
    top_n_freqs = [freq[1] for freq in sorted_word_tuples[:n]]

    # Set colors for bars
    bar_colors = ['red' if word in highlight_words else 'skyblue' for word in top_n_words]

    # Clear the current output
    clear_output(wait=True)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.bar(top_n_words, top_n_freqs, color=bar_colors)
    
    if logscale:
        plt.yscale('log')

    # Apply the custom font to the x-axis labels
    rotation = 90 if title == "words" else 0
    
    plt.xticks(rotation=rotation, fontproperties=prop)

    plt.xlabel(title, fontproperties=prop)
    plt.ylabel('Frequency')
    plt.ylim(0, top_n_freqs[0])
    plt.title(f'Top {n} {title} and Their Frequencies', fontproperties=prop)

    if hide_xticks:
        plt.xticks([])  # Hide x-axis ticks

    plt.tight_layout()
    
    if animate:
        display(plt.gcf())
    
    plt.close()

# ------------------------------------------------------

def plot_cumulative_coverage(sorted_tuples, limit=1000, title="words"):
    
    frequency_values = [f for _, f in sorted_tuples]
    
    if limit is not None:
        frequency_values = frequency_values[:limit]

    cumulative_sum = [sum(frequency_values[:i+1]) for i in range(len(frequency_values))]

    total_sum = sum(frequency_values)

    cumulative_percentage = [(sum_values / total_sum) * 100 for sum_values in cumulative_sum]

    # Plotting
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(frequency_values) + 1), cumulative_percentage, marker='o') # , linestyle='-')

    plt.xlabel(f'Number of {title}')
    plt.ylabel('Cumulative Percentage of Total Frequency')
    plt.title(f'Cumulative Distribution of {title} Frequencies')
    plt.grid(True)
    # plt.xticks(range(1, len(frequency_values) + 1))
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# ------------------------------------------------------

# def interactive_line_plot(numbers_list, xlabel="", ylabel="", title="", ylim=None):
        
#     def line_plot(numbers_list, x_range, xlabel='X-axis', ylabel='Y-axis', title='Line Plot', rolling_mean_k=1, ylim=None):
        
#         selected_numbers = numbers_list[:x_range]
        
#         if rolling_mean_k > 1:
#             selected_numbers = utils.rolling_mean(selected_numbers, rolling_mean_k)
        
#         plt.plot(selected_numbers)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.title(title)
#         plt.grid(True)
#         if ylim is None:
#             plt.ylim((0, max(selected_numbers)))
#         else:
#             plt.ylim(ylim)
            
#         plt.show()
    
#     # Create widgets
#     x_range_slider = widgets.IntSlider(value=len(numbers_list), min=2, max=len(numbers_list), step=1, description='X Range:')
#     rolling_mean_slider = widgets.IntSlider(value=1, min=1, max=10, step=1, description='Rolling Mean K:')

#     interactive_plot = widgets.interactive(line_plot, 
#                                         numbers_list=widgets.fixed(numbers_list),
#                                         x_range=x_range_slider,
#                                         xlabel=widgets.fixed(xlabel),
#                                         ylabel=widgets.fixed(ylabel),
#                                         title=widgets.fixed(title),
#                                         rolling_mean_k=rolling_mean_slider,
#                                         ylim=widgets.fixed(ylim))
#     display(interactive_plot)
    
# ------------------------------------------------------


def interactive_line_plot(numbers_lists, legend_labels, xlabel="", ylabel="", title="", ylim=None):
        
    def line_plot(numbers_lists, x_range, xlabel='X-axis', ylabel='Y-axis', title='Line Plot', rolling_mean_k=1, ylim=None):
        plt.figure(figsize=(10, 6))

        for numbers, label in zip(numbers_lists, legend_labels):
            selected_numbers = numbers[:x_range]
            if rolling_mean_k > 1:
                selected_numbers = utils.rolling_mean(selected_numbers, rolling_mean_k)
            plt.plot(selected_numbers, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        if ylim is None:
            plt.ylim(0, max(max(numbers) for numbers in numbers_lists))
        else:
            plt.ylim(ylim)
        plt.legend()
        plt.show()
    
    max_val = max(len(list_) for list_ in numbers_lists)        
    
    x_range_slider = widgets.IntSlider(value=len(numbers_lists[0]), min=2, max=max_val, step=1, description='X Range:')
    rolling_mean_slider = widgets.IntSlider(value=1, min=1, max=10, step=1, description='Rolling Mean K:')

    interactive_plot = widgets.interactive(line_plot, 
                                        numbers_lists=widgets.fixed(numbers_lists),
                                        x_range=x_range_slider,
                                        xlabel=widgets.fixed(xlabel),
                                        ylabel=widgets.fixed(ylabel),
                                        title=widgets.fixed(title),
                                        rolling_mean_k=rolling_mean_slider,
                                        ylim=widgets.fixed(ylim))
    display(interactive_plot)
