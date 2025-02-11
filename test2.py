import matplotlib.pyplot as plt
import numpy as np

def visualize_movements(movements, start, grid):
    x, y = start
    path_x = [x]
    path_y = [y]

    move_map = {
        1: (-2, 0),  # Up
        2: (0, 2),   # Right
        3: (2, 0),   # Down
        4: (0, -2)   # Left
    }

    for move in movements:
        dx, dy = move_map[move]
        x += dx
        y += dy
        path_x.append(x)
        path_y.append(y)

    plt.figure(figsize=(10, 8))

    # Create grid visualization
    grid_array = np.array(grid)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            value = grid[r][c]
            color = 'white'
            if value == 1:  # Start
                color = 'green'
            elif value == 2:  # End
                color = 'red'
            elif value == 3:  # Barrier
                color = 'black'
            elif value == 4:  # Gate
                color = 'blue'
            plt.gca().add_patch(plt.Rectangle((c, r), 1, 1, color=color, ec='gray'))

    # Plot the path
    plt.plot(path_y, path_x, marker='o', linestyle='-', color='orange', label='Path')
    plt.scatter(path_y[0], path_x[0], color='green', s=100, label='Start')  # Start point
    plt.scatter(path_y[-1], path_x[-1], color='red', s=100, label='End')    # End point

    # Format plot
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.xticks(range(len(grid[0]) + 1))
    plt.yticks(range(len(grid) + 1))
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('Path Visualization with Grid')
    plt.legend()
    plt.show()


# Example usage
grid = [
    [1, 0, 4, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 9, 0, 9, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 9, 0, 9, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 9, 0, 9, 0, 9, 0],
    [0, 0, 4, 0, 0, 0, 4, 0, 2]
]

movements = [2, 3, 2, 3, 2, 2, 1]
start_position = (0, 0)
visualize_movements(movements, start_position, grid)
