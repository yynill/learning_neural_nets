import json
import os
import pygame
import numpy as np


def load_trainingData():
    try:
        if os.path.isfile('training_data.json') and os.path.getsize('training_data.json') > 0:
            with open('training_data.json', 'r') as json_file:
                training_data = json.load(json_file)
        else:
            training_data = []

        return training_data
    except:
        print("training_data file not found!")


def extract_trainingData(training_data):
    sample_count = len(training_data)
    number_value_count = {}
    for entry in training_data:
        number_value = entry['number_value']
        number_value_count[number_value] = number_value_count.get(
            number_value, 0) + 1

    print("\nTotal samples:", sample_count)
    sorted_number_value_count = sorted(
        number_value_count.items(), key=lambda x: x[0])
    for number_value, count in sorted_number_value_count:
        print(f"| {number_value} samples: {count}")
    print("\n")


def extract_sample(training_data, s, num_of_tiles=28):
    entry = training_data[s-1]

    input_id = entry['input_id']
    number_value = entry['number_value']
    tile_values = entry['tile_values']

    # Ensure tile_values array has 784 elements
    if len(tile_values) != num_of_tiles * num_of_tiles:
        print("Tile values array must have 784 elements")

    # Convert tile_values to a 28x28 array 0-255
    arr_2d = np.reshape(tile_values, (num_of_tiles, num_of_tiles))
    scaled_array = np.array(arr_2d) * 255  # Scaling to 0-255
    scaled_array = scaled_array.astype(int)  # Converting to integers

    # Converting to RGB tuples
    rgb_array = [[(0, 0, 0) if value == 0 else (255, 255, 255) if value == 255 else (
        int(value), int(value), int(value)) for value in row] for row in scaled_array]

    return rgb_array


def draw_window(window, board_colors):
    window.fill((0, 0, 0))  # Clear the window

    for row in range(num_of_tiles):
        for col in range(num_of_tiles):
            pygame.draw.rect(
                window, board_colors[row][col], (col * tile_size, row * tile_size, tile_size, tile_size))

    pygame.display.flip()  # Update the display

###################################
###################################


num_of_tiles = 28
tile_size = 10
board_height = num_of_tiles * tile_size
board_width = board_height

training_data = load_trainingData()
pygame.init()

window = pygame.display.set_mode(
    (board_width, board_height))
pygame.display.set_caption('Number')

# Initialize the board with all tiles set to black
board_colors = [[(0, 0, 0)
                for _ in range(num_of_tiles)]
                for _ in range(num_of_tiles)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    user_input = input(">>: ")
    if user_input.lower() == "quit":
        running = False
    elif user_input.lower() == "samples":
        extract_trainingData(training_data)
    elif user_input[:4].lower() == "load":
        number = int(user_input[5:])
        board_colors = extract_sample(training_data, number)

        # Draw the window
        window.fill((0, 0, 0))
        draw_window(window, board_colors)

    else:
        print("Invalid input.")

pygame.quit()
