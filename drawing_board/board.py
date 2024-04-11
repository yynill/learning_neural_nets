import pygame
import png_save

# Constants for the game board
num_of_tiles = 28
tile_size = 10
board_height = num_of_tiles * tile_size
board_width = board_height
grayscale_increment = 50

# Initialize Pygame
pygame.init()

# Create the game window
window = pygame.display.set_mode((board_width, board_height))
pygame.display.set_caption('Board')

# Initialize the board with all tiles set to black
board_colors = [[(0, 0, 0)
                for _ in range(num_of_tiles)]
                for _ in range(num_of_tiles)]


def reset():
    global board_colors
    board_colors = [[(0, 0, 0)
                    for _ in range(num_of_tiles)]
                    for _ in range(num_of_tiles)]


def draw_window():
    for row in range(num_of_tiles):
        for col in range(num_of_tiles):
            pygame.draw.rect(window, board_colors[row][col], (col * tile_size,
                             row * tile_size, tile_size, tile_size))


def paint_tile(mouse_x, mouse_y):
    row = mouse_y // tile_size
    col = mouse_x // tile_size

    # surrounding pixels
    for i, j in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + i, col + j
        if 0 <= new_row < num_of_tiles and 0 <= new_col < num_of_tiles:
            current_gray = board_colors[new_row][new_col][0]
            new_gray = min(current_gray + grayscale_increment, 255)
            board_colors[new_row][new_col] = (new_gray, new_gray, new_gray)


def board():
    running = True
    drawing = False
    clock = pygame.time.Clock()

    while running:
        clock.tick(100)
        for event in pygame.event.get():
            # quit
            if event.type == pygame.QUIT:
                running = False

            # draw
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

            elif event.type == pygame.KEYDOWN:
                # reset
                if event.key == pygame.K_x:
                    reset()

                # save
                elif event.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9):
                    key_as_int = event.key - pygame.K_0
                    png_save.save(board_colors, key_as_int)
                    print(f'Saved as - {key_as_int} -')
                    reset()

        window.fill((0, 0, 0))

        if drawing:
            mouse_x, mouse_y = event.pos
            paint_tile(mouse_x, mouse_y)

        draw_window()

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    board()
