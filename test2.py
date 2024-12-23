import matplotlib.pyplot as plt
import numpy as np

# Define the grid (with 1 as start, 2 as end, 3 as gates, 4 as barriers)
grid = [
    [1, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 4, 0, 4, 0, 4, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 4, 0, 4, 0, 4, 0],
    [0, 0, 3, 0, 0, 0, 3, 0, 0],
    [0, 4, 0, 4, 0, 4, 0, 4, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 3],
]

# Set up the figure and axes
fig, ax = plt.subplots()

# Loop through the grid and draw the actual tiles (even-numbered coordinates)
for i in range(0, len(grid), 2):
    for j in range(0, len(grid[0]), 2):
        if grid[i][j] == 1:  # Start point
            ax.add_patch(plt.Rectangle((j, i), 2, 2, edgecolor='black', facecolor='green'))
        elif grid[i][j] == 2:  # End point
            ax.add_patch(plt.Rectangle((j, i), 2, 2, edgecolor='black', facecolor='red'))
        elif grid[i][j] == 3:  # Gate
            ax.add_patch(plt.Rectangle((j, i), 2, 2, edgecolor='black', facecolor='blue'))
        else:  # Empty tile
            ax.add_patch(plt.Rectangle((j, i), 2, 2, edgecolor='black', facecolor='white'))

# Draw the barriers as lines between tiles (check odd coordinates for barriers)
for i in range(1, len(grid), 2):  # Vertical barriers between rows
    for j in range(0, len(grid[0]), 2):
        if grid[i][j] == 4:  # Barrier between two rows
            ax.plot([j, j+2], [i-1, i-1], color='black', linewidth=4)  # Draw horizontal barrier

for i in range(0, len(grid), 2):  # Horizontal barriers between columns
    for j in range(1, len(grid[0]), 2):
        if grid[i][j] == 4:  # Barrier between two columns
            ax.plot([j-1, j-1], [i, i+2], color='black', linewidth=4)  # Draw vertical barrier

# Set axis limits and aspect ratio
ax.set_xlim(0, 9)
ax.set_ylim(7, 0)
ax.set_aspect('equal')

# Hide the axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])

plt.show()
