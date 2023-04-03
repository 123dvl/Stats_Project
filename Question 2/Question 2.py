import math
import matplotlib.pyplot as plt
import re



def stem_and_leaf_plot(numbers):
    numbers.sort()
    print("\n-------Data in Order-------\n")
    print(numbers)
    stems = []
    leaves = []
    print("\n-------Stem and Leaf-------\n")
    for num in numbers:
        stem, leaf = divmod(int(num * 10), 10)
        stems.append(stem)
        leaves.append(round(leaf))
    stem_leaf_pairs = list(zip(stems, leaves))
    stem_leaf_dict = {}
    for pair in stem_leaf_pairs:
        stem = pair[0]
        leaf = pair[1]
        if stem not in stem_leaf_dict:
            stem_leaf_dict[stem] = []
        stem_leaf_dict[stem].append(leaf)
    for stem, leaves in sorted(stem_leaf_dict.items()):
        print(f"{stem} | {' '.join(map(str, sorted(leaves)))}")


def count_numbers(numbers):
    return len(numbers)


def create_histogram(numbers, num_intervals):
    min_num = min(numbers)
    max_num = max(numbers)
    range_nums = max_num - min_num
    interval_size = range_nums / num_intervals

    boundaries = []
    lower_boundary = min_num
    for i in range(num_intervals):
        upper_boundary = lower_boundary + interval_size
        boundaries.append((lower_boundary, upper_boundary))
        lower_boundary = upper_boundary

    frequencies = [0] * num_intervals
    for num in numbers:
        interval_index = int((num - min_num) // interval_size)
        if interval_index == num_intervals:
            interval_index -= 1
        frequencies[interval_index] += 1

    total = len(numbers)
    relative_frequencies = [freq / total for freq in frequencies]
    cumulative_frequencies = [sum(frequencies[:i+1]) for i in range(num_intervals)]
    cumulative_relative_frequencies = [cum_freq / total for cum_freq in cumulative_frequencies]

    intervals = [(start, end) for start, end in boundaries]

    return boundaries, intervals, frequencies, relative_frequencies, cumulative_frequencies, cumulative_relative_frequencies

def draw_histogram(boundaries, frequencies):
    print("\n-------Frequency and Relative Frequency-------\n")
    for boundary, freq in zip(boundaries, frequencies):
        start = boundary[0]
        end = boundary[1]
        print(f"{start:.2f} to {end:.2f} | {freq:3} | {freq/sum(frequencies):.3f}")



def create_ogive(numbers, num_intervals):
    class_intervals = [(min(numbers) + i * (max(numbers) - min(numbers)) / num_intervals,
                        min(numbers) + (i + 1) * (max(numbers) - min(numbers)) / num_intervals) for i in
                       range(num_intervals)]
    frequencies = [len([num for num in numbers if class_interval[0] <= num < class_interval[1]]) for class_interval in
                   class_intervals]
    rel_frequencies = [frequency / len(numbers) for frequency in frequencies]
    cum_rel_frequencies = [sum(rel_frequencies[:i + 1]) for i in range(len(rel_frequencies))]

    fig, ax = plt.subplots()
    ax.plot([class_intervals[0][0]] + [class_interval[1] for class_interval in class_intervals],
            [0] + cum_rel_frequencies)
    ax.set_xlabel('Values')
    ax.set_ylabel('Cumulative relative frequency')
    ax.set_title('Ogive')
    plt.show()


def draw_relative_frequency_plot(intervals, relative_frequencies):
    interval_centers = [(start + end) / 2 for start, end in intervals]

    plt.bar(interval_centers, relative_frequencies, width=intervals[1][0] - intervals[0][0], align='center', alpha=0.7,
            label='Histogram')
    plt.plot(interval_centers, relative_frequencies, marker='o', linestyle='-', color='r', label='Polygon')
    plt.xlabel('Intervals')
    plt.ylabel('Relative Frequency')
    plt.legend()
    plt.show()

def draw_cumulative_relative_frequency_plot(intervals, cumulative_relative_frequencies):
    fig, ax = plt.subplots()
    interval_centers = [(start + end) / 2 for start, end in intervals]
    ax.hist(interval_centers, bins=len(intervals), weights=cumulative_relative_frequencies, cumulative=True, density=True, edgecolor="black")
    ax.set_xlabel("Intervals")
    ax.set_ylabel("Cumulative Relative Frequencies")
    ax.set_title("Cumulative Relative Frequencies Histogram")
    plt.show()

def draw_ogive_plot(intervals, cumulative_frequencies):
    cumulative_percentages = [freq / sum(cumulative_frequencies) * 100 for freq in cumulative_frequencies]
    cumulative_percentages.insert(0, 0)

    x_values = [interval[0] for interval in intervals]
    x_values.insert(0, intervals[0][0])

    plt.plot(x_values, cumulative_percentages, 'ro-')
    plt.title('Ogive Plot')
    plt.xlabel('Intervals')
    plt.ylabel('Cumulative Percentages')
    plt.xticks(x_values)
    plt.show()


def five_num_summary(numbers):
    sorted_numbers = sorted(numbers)
    n = len(numbers)
    Q1 = sorted_numbers[n // 4]
    Q2 = sorted_numbers[n // 2]
    Q3 = sorted_numbers[n // 4 * 3]
    minimum = sorted_numbers[0]
    maximum = sorted_numbers[-1]
    print("\n-------5 Number Summary + IQR + Outliers-------\n")
    print("Minimum:", minimum)
    print("Q1:", Q1)
    print("Median (Q2):", Q2)
    print("Q3:", Q3)
    print("Maximum:", maximum)
    IQR = Q3 - Q1
    print("IQR:",IQR)
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    outliers = [x for x in sorted_numbers if x < lower_fence or x > upper_fence]
    print("Outliers:", outliers)
    return (minimum, Q1, Q2, Q3, maximum)


    # Return the 5-number summary
    return min_val, q1, median, q3, max_val

def draw_box_plot(numbers):
    # Find the 5-number summary
    min_val, q1, median, q3, max_val = five_num_summary(numbers)

    # Create a box plot using matplotlib
    fig, ax = plt.subplots()
    ax.boxplot(numbers, vert=False, showfliers=True, whis=1.5)
    ax.set_xlabel('Value')
    ax.set_title('Box plot')

    # Show the plot
    plt.show()


def calculate_statistics(numbers):
    print("\n-------Mean Median Mode Variance Standard Deviation Range-------\n")
    # Mean
    mean = sum(numbers) / len(numbers)

    # Median
    numbers_sorted = sorted(numbers)
    middle = len(numbers_sorted) // 2
    if len(numbers_sorted) % 2 == 0:
        median = (numbers_sorted[middle - 1] + numbers_sorted[middle]) / 2
    else:
        median = numbers_sorted[middle]

    # Mode
    mode = max(set(numbers), key=numbers.count)

    # Variance
    variance = sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)

    # Standard deviation
    standard_deviation = variance ** 0.5

    # Range
    range_val = max(numbers) - min(numbers)

    # Print the results
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Variance: {variance}")
    print(f"Standard deviation: {standard_deviation}")
    print(f"Range: {range_val}")



def read_numbers_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    numbers = []
    for line in lines:
        line_numbers = line.split()
        for num in line_numbers:
            try:
                numbers.append(float(num))
            except ValueError:
                pass
    return numbers



if __name__ == "__main__":
    input_file = "input_data.txt"
    numbers = read_numbers_from_file(input_file)

    print(f"Number of input values: {count_numbers(numbers)}")

    stem_and_leaf_plot(numbers)

    num_intervals = 6  # Set the predetermined class interval here
    boundaries, intervals, frequencies, relative_frequencies, cumulative_frequencies, cumulative_relative_frequencies = create_histogram(
        numbers, num_intervals)
    draw_histogram(boundaries, frequencies)
    draw_relative_frequency_plot(intervals, relative_frequencies)
    draw_cumulative_relative_frequency_plot(intervals, cumulative_relative_frequencies)
    draw_ogive_plot(intervals, cumulative_frequencies)
    draw_box_plot(numbers)
    calculate_statistics(numbers)
