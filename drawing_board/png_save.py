import json
import os


def save(board_colors, key_save):
    tile_colors = []
    for row in board_colors:
        for col in row:
            gray_scale_value = round((col[0]/255), 2)
            tile_colors.append(gray_scale_value)

    try:
        if os.path.isfile('training_data.json') and os.path.getsize('training_data.json') > 0:
            with open('training_data.json', 'r') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []
    except FileNotFoundError:
        existing_data = []

    next_input_id = len(existing_data) + 1
    data_to_save = {
        "input_id": next_input_id,
        "number_value": key_save,
        "tile_values": tile_colors
    }

    existing_data.append(data_to_save)

    with open('training_data.json', 'w') as json_file:
        json.dump(existing_data, json_file)
