import numpy as np

def DFS(maze, path):
    S = maze.sense_robot()
    E = maze.destination
    if S == E:
        return path
    
    height, weight, _ = maze.maze_data.shape
    is_visit_m = np.zeros((height, weight), dtype=np.int)  # Mark positions visited in the maze
    
    maze.reset_robot()
    for i in path:
        is_visit_m[maze.sense_robot()] = 1
        maze.move_robot(i)  # Mark positions visited while moving from S to the current position
    
    choices = maze.can_move_actions(S)  # Get available choices from the current position
    for i in choices:
        maze.move_robot(i)  # Try each possible direction
        
        if is_visit_m[maze.sense_robot()] == 1:  # If already visited, backtrack to the starting position
            maze.robot["loc"] = S
            continue
        
        path.append(i)  # Extend the path
        new = DFS(maze, path)  # Recursively search
        
        if maze.sense_robot() == E:  # If reached the end point, update the path
            path = new
            break
        
        del(path[-1])  # If not at the end point, restore the current path
        maze.robot["loc"] = S  # Return to the starting position
    
    return path

def my_search(maze):
    """
    Choose depth-first search or A* algorithm implementation
    :param maze: Maze object
    :return: Path to reach the destination point, e.g., ["u", "u", "r", ...]
    """
    path = DFS(maze, [])
    return path
