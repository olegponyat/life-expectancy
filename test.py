import heapq
import matplotlib.pyplot as plt
import numpy as np

# 1 = start, 2 = end, 3 = gates, 4 = barriers
grid = [
    [0, 0, 2, 4, 3, 4, 0,],
    [0, 0, 4, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0,],
    [4, 0, 0, 0, 4, 0, 0,],
    [3, 0, 0, 4, 0, 0, 3,],
    [0, 0, 4, 0, 0, 0, 4,],
    [0, 0, 1, 0, 0, 0, 0,],
    
]

def find_positions(grid):
    start, end = None, None
    gates = []
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                start = (r, c)
            elif grid[r][c] == 2:
                end = (r, c)
            elif grid[r][c] == 3:
                gates.append((r, c))
    return start, end, gates

def calculate_shortest_path(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    start_score = {start: 0}
    final_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            neighbor = (current[0] + dr, current[1] + dc)
            barrier_check = (current[0] + dr // 2, current[1] + dc // 2)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[barrier_check[0]][barrier_check[1]] != 4:
                relative_start_score = start_score[current] + 1
                if relative_start_score < start_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    start_score[neighbor] = relative_start_score
                    final_score[neighbor] = relative_start_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (final_score[neighbor], neighbor))

    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path

def calculate_gates_and_end(grid, start, end, gates):
    current_position = start
    total_path = []
    visited_gates = set()

    while len(visited_gates) < len(gates):
        nearest_gate, shortest_path = None, None
        shortest_distance = float('inf')

        for gate in gates:
            if gate not in visited_gates:
                path_to_gate = calculate_shortest_path(grid, current_position, gate)
                distance_to_gate = len(path_to_gate)
                if distance_to_gate < shortest_distance:
                    nearest_gate = gate
                    shortest_distance = distance_to_gate
                    shortest_path = path_to_gate

        if nearest_gate:
            total_path.extend(shortest_path)
            visited_gates.add(nearest_gate)
            current_position = nearest_gate

    final_path = calculate_shortest_path(grid, current_position, end)
    total_path.extend(final_path)

    return total_path


def visualize_grid(grid, path, start, end, gates):
    color_map = np.zeros((len(grid), len(grid[0])), dtype=int)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                color_map[r][c] = 1
            elif grid[r][c] == 2:
                color_map[r][c] = 2
            elif grid[r][c] == 3:
                color_map[r][c] = 3
            elif grid[r][c] == 4:
                color_map[r][c] = 4
    plt.figure(figsize=(12, 6))
    plt.imshow(color_map, cmap='tab10', vmin=0, vmax=4)
    path_y, path_x = zip(*path)
    plt.plot(path_x, path_y, 'r-', linewidth=2)
    first_gate = path[0] if path else start
    plt.plot([start[1], first_gate[1]], [start[0], first_gate[0]], 'r-', linewidth=2)
    for (r, c) in gates:
        plt.plot(c, r, 'bo', markersize=12)
    plt.plot(start[1], start[0], 'go', markersize=12)
    plt.plot(end[1], end[0], 'ro', markersize=12)
    total_moves = len(path) 
    plt.text(len(grid[0]), len(grid) / 2, f'Total Moves: {total_moves}', fontsize=12, verticalalignment='center', horizontalalignment='left', color='black')
    plt.xticks(np.arange(len(grid[0])), np.arange(1, len(grid[0]) + 1))
    plt.yticks(np.arange(len(grid)), np.arange(1, len(grid) + 1))
    plt.grid(which='both')
    plt.title("Robot Tour Visualization")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

start, end, gates = find_positions(grid)
path = calculate_gates_and_end(grid, start, end, gates)
visualize_grid(grid, path, start, end, gates)

print("Start:", start)
print("End:", end)
print("Gates:", gates)
print("Most efficient path:", path)