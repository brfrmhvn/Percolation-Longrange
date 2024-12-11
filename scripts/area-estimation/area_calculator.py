import numpy as np

# Determine whether it is a connector
def is_connector(grid, x, y):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    count_zeros = 0
    for dx, dy in directions:
        nx = (x + dx) % len(grid)
        ny = (y + dy) % len(grid[0])
        if grid[nx][ny] == 0:
            count_zeros += 1
    return count_zeros >= 3

def dfs(grid, x, y, visited):
    stack = [(x % len(grid), y % len(grid[0]))]
    area = 0
    while stack:
        x, y = stack.pop()
        x = x % len(grid)
        y = y % len(grid[0])
        if visited[x][y]:
            continue
        if grid[x][y] == 1:
            continue
        visited[x][y] = True
        if grid[x][y] == 0:
            area += 1
        elif grid[x][y] == 3:
            if not is_connector(grid, x, y):
                continue
        else:
            continue
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))
    return area

# Calculate the area 
def calculate_nucleus_area_with_connectors(grid):
    visited = np.zeros_like(grid, dtype=bool)
    areas = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0 and not visited[i][j]:
                area = dfs(grid, i, j, visited)
                areas.append(area)
    return areas