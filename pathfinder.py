import numpy as np
from heapq import heappop, heappush

def a_star(map_data, start, end, weights):
    """A* algorithm with terrain weights"""
    open_set = [(0, *start)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        current = heappop(open_set)[1:]
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if (0 <= neighbor[0] < map_data.shape[0] and 
                0 <= neighbor[1] < map_data.shape[1]):
                terrain = map_data[neighbor]
                cost = weights.get(terrain, 1)
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(end))
                    heappush(open_set, (priority, *neighbor))
    return []
