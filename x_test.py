import numpy as np


def convert_array_to_whole_numbers(arr):
    return np.round(arr * 255).astype(np.uint8)


# Example usage:
array = np.array([[0, 0, 0.1],
                  [0.2, 0.3, 0.4]])
converted_array = convert_array_to_whole_numbers(array)
print(converted_array)
