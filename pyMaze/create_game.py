import pygame
import numpy as np
from solve_game import *

width, height = 800, 600
block_size = 25
color = (0, 0 , 0)

cols = int(height / block_size)
rows = int(width / block_size)


def create_maze(surface, wall, prev_maze = None):
    if prev_maze is None:
        maze = np.random.randint(0, high = 2, size = (rows, cols), dtype = 'int')
    else:
        maze = prev_maze
    ones = np.argwhere(maze == 1)
    for one in ones:
        x, y = one
        surface.blit(wall, (x * block_size, y * block_size))

    return maze


pygame.init()
_display_surface = pygame.display.set_mode((width, height))
pygame.display.set_caption('Maze_2D')
clock = pygame.time.Clock()
_running = True

_display_surface.fill((100, 20, 40))
_background_img = pygame.image.load('images/background.jpg')
_background_img.set_alpha(100)
_player_img = pygame.image.load('images/player.png')
_wall_img = pygame.image.load('images/wall.png')
_dead_end = pygame.image.load('images/dead_end.jpg')
_target_img = pygame.image.load('images/target_2.png')

# maze = None
maze = np.zeros((rows, cols))
# maze = np.random.randint(0, high = 2, size = (rows, cols), dtype = 'int')

while _running:
    for event in pygame.event.get():

        # print(event)

        if event.type == pygame.QUIT:
            _running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            x = int(x / block_size)
            y = int(y / block_size)
            maze[x, y] = 1

            for y1 in range(cols):
                for x1 in range(rows):
                    rect = pygame.Rect(x1*block_size, y1*block_size, block_size-1, block_size-1)
                    pygame.draw.rect(_display_surface, color, rect)

            _display_surface.blit(_background_img, (0, 0))

            maze = create_maze(_display_surface, _wall_img, prev_maze = maze)

    pygame.display.update()
    clock.tick(30)

start = node(2, 2)
target = node(20, 0)
trace = A_star(maze, start, target, rows, cols)
if trace is None:
    _display_surface.blit(_dead_end, (0, 0))

else:
    trace.reverse()

_running = True

while _running:
    for path in trace:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                x = int(x / block_size)
                y = int(y / block_size)
                start = target
                target = node(x, y)
                trace = A_star(maze, start, target, rows, cols)
                if trace is None:
                    print("no path found")
                else:
                    trace.reverse()
                    # continue

        for y1 in range(cols):
            for x1 in range(rows):
                rect = pygame.Rect(x1*block_size, y1*block_size, block_size-1, block_size-1)
                pygame.draw.rect(_display_surface, color, rect)

        _display_surface.blit(_background_img, (0, 0))

        maze = create_maze(_display_surface, _wall_img, prev_maze = maze)

        _display_surface.blit(_target_img, (target.x * block_size, target.y * block_size))

        # node.node_print()

        if maze[path.x, path.y] != 1:
            _display_surface.blit(_player_img, (path.x* block_size, path.y* block_size))


        pygame.display.update()
        clock.tick(30)

pygame.quit()
quit()