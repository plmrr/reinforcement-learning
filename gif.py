import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize(state, turn, frame_path, board_size):
    color_map = {
        1: [172, 44, 7],  # Settler
        2: [3, 47, 25],  # Scout
        3: [255, 199, 88],  # Basic city
        4: [102, 51, 0],  # Wooden city
        5: [161, 157, 148],  # Iron city
        6: [133, 94, 66],  # Wood
        7: [78, 79, 85],  # Iron
        8: [215, 183, 64]  # Gold
    }

    labels = {
        1: "Settler",
        2: "Scout",
        3: "City",
        4: "W_City",
        5: "I_City",
        6: "Wood",
        7: "Iron",
        8: "Gold"
    }

    board = np.zeros((board_size, board_size, 3))
    text = np.full((board_size, board_size), "", dtype='U20')
    for unit in state['units']:
        if unit[0]:  # Settler
            board[unit[1][0], unit[1][1]] = color_map[1]
            if text[unit[1][0], unit[1][1]] != "":
                text[unit[1][0], unit[1][1]] += "+" + labels[1]
            else:
                text[unit[1][0], unit[1][1]] = labels[1]
        else:  # Scout
            board[unit[1][0], unit[1][1]] = color_map[2]
            if text[unit[1][0], unit[1][1]] != "":
                text[unit[1][0], unit[1][1]] += "+" + labels[2]
            else:
                text[unit[1][0], unit[1][1]] = labels[2]

    for city in state['basic_cities']:
        board[city[0], city[1]] = color_map[3]
        if text[city[0], city[1]] != "":
            text[city[0], city[1]] += "+" + labels[3]
        else:
            text[city[0], city[1]] = labels[3]

    for city in state['wooden_cities']:
        board[city[0], city[1]] = color_map[4]
        if text[city[0], city[1]] != "":
            text[city[0], city[1]] += "+" + labels[4]
        else:
            text[city[0], city[1]] = labels[4]

    for city in state['iron_cities']:
        board[city[0], city[1]] = color_map[5]
        if text[city[0], city[1]] != "":
            text[city[0], city[1]] += "+" + labels[5]
        else:
            text[city[0], city[1]] = labels[5]

    for resource in state['resources'].keys():
        resource_type = state['resources'][resource]
        if resource_type == 'wood':
            board[resource[0], resource[1]] = color_map[6]
            text[resource[0], resource[1]] = labels[6]
        elif resource_type == 'iron':
            board[resource[0], resource[1]] = color_map[7]
            text[resource[0], resource[1]] = labels[7]
        elif resource_type == 'gold':
            board[resource[0], resource[1]] = color_map[8]
            text[resource[0], resource[1]] = labels[8]

    board[board.sum(axis=2) == 0] = [96, 221, 73]

    board = board / 255.0  # normalize rgb

    fig, ax = plt.subplots()
    ax.imshow(board)

    for i in range(board_size):
        for j in range(board_size):
            ax.text(j, i, text[i, j], ha='center', va='center', color='black', fontsize=8)

    ax.set_xticks(np.arange(-.5, board_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, board_size, 1), minor=True)

    ax.grid(which='minor', color='black', linewidth=2)
    plt.title(f'Turn: {turn}, gold: {state["gold"]}')
    plt.savefig(f'{frame_path}/frame_{str(turn).zfill(5)}.png')
    plt.close()

def create_gif(image_folder, output_file, duration=500):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filepath = os.path.join(image_folder, filename)
            try:
                image = Image.open(filepath)
                images.append(image)
            except IOError as e:
                print(f"Error opening image {filepath}: {e}")

    if images:
        images[0].save(output_file, format='GIF',
                       append_images=images[1:],
                       save_all=True,
                       duration=duration,
                       loop=0)
    else:
        print("No images found to create GIF")
