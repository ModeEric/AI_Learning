import random

# Grid configuration
gridsize = (10, 10)

# Define states as a grid of (x, y) positions
states = [[(i, j) for j in range(gridsize[1])] for i in range(gridsize[0])]

# Define possible actions: right, down, up, left
actions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
num_actions = len(actions)

# Q-learning parameters
gamma = 0.9      # Discount factor
alpha = 0.2      # Learning rate
epsilon = 1.0    # Exploration rate
min_epsilon = 0.1
epsilon_decay = 0.995

# Starting and ending positions
startingpos = (gridsize[0] // 2, gridsize[1] // 2)  # Center of the grid
endingpos = (gridsize[0] - 1, gridsize[1] - 1)      # Bottom-right corner

# Function to check if a move is invalid (i.e., goes out of bounds)
def isinvalid(move, pos):
    new_x = pos[0] + move[0]
    new_y = pos[1] + move[1]
    if new_x < 0 or new_x >= gridsize[0] or new_y < 0 or new_y >= gridsize[1]:
        return True
    return False

# Initialize Q-table with zeros
Q = [[[0 for _ in range(num_actions)] for _ in range(gridsize[1])] for _ in range(gridsize[0])]

# Number of episodes for training
num_episodes = 500

for episode in range(num_episodes):
    pos = startingpos
    step = 0
    while pos != endingpos and step < 1000:  # Limit steps to prevent infinite loops
        step += 1
        x, y = pos

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_index = random.randint(0, num_actions - 1)
        else:
            # Select the action with the highest Q-value for the current state
            max_q = max(Q[x][y])
            # In case of multiple actions with the same max Q-value, choose randomly among them
            best_actions = [i for i, q in enumerate(Q[x][y]) if q == max_q]
            action_index = random.choice(best_actions)

        move = actions[action_index]

        # Check if the move is invalid
        if isinvalid(move, pos):
            reward = -1  # Penalty for invalid move
            next_pos = pos  # Stay in the same position
        else:
            # Calculate new position
            next_pos = (pos[0] + move[0], pos[1] + move[1])
            # Reward can be based on the negative distance to the goal (encouraging closer moves)
            distance_current = (endingpos[0] - pos[0])**2 + (endingpos[1] - pos[1])**2
            distance_new = (endingpos[0] - next_pos[0])**2 + (endingpos[1] - next_pos[1])**2
            reward = (distance_current - distance_new) / (gridsize[0]**2 + gridsize[1]**2)
            if next_pos == endingpos:
                reward = 1  # Reward for reaching the goal

        # Update Q-value using the Q-learning update rule
        old_q = Q[x][y][action_index]
        if next_pos == endingpos:
            max_future_q = 0  # No future rewards since it's the terminal state
        else:
            max_future_q = max(Q[next_pos[0]][next_pos[1]])
        Q[x][y][action_index] = old_q + alpha * (reward + gamma * max_future_q - old_q)

        # Move to the next position
        pos = next_pos

    # Decay epsilon to reduce exploration over time
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    else:
        epsilon = min_epsilon

    # Optional: Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed. Epsilon: {epsilon:.4f}")

# After training, print the Q-table
# Note: Printing the entire Q-table for a 10x10 grid can be overwhelming.
# Instead, you might want to visualize it or inspect specific states.

# Example: Print Q-values for the starting position
print(f"Q-values at starting position {startingpos}:")
for idx, action in enumerate(actions):
    print(f"  Action {action}: {Q[startingpos[0]][startingpos[1]][idx]:.2f}")

# Optional: Function to derive the optimal path from the starting position
def get_optimal_path(Q, start, end):
    path = [start]
    pos = start
    while pos != end:
        x, y = pos
        # Choose the action with the highest Q-value
        max_q = max(Q[x][y])
        best_actions = [i for i, q in enumerate(Q[x][y]) if q == max_q]
        action_index = random.choice(best_actions)
        move = actions[action_index]
        next_pos = (pos[0] + move[0], pos[1] + move[1])

        # If the move is invalid or leads to the same position, stop to prevent infinite loop
        if isinvalid(move, pos) or next_pos == pos:
            print("No valid path found.")
            break

        path.append(next_pos)
        pos = next_pos

        # Safety check to prevent infinite loops
        if len(path) > gridsize[0] * gridsize[1]:
            print("Path is too long, stopping.")
            break

    return path

# Get and print the optimal path
optimal_path = get_optimal_path(Q, startingpos, endingpos)
print("Optimal Path:")
for step, position in enumerate(optimal_path):
    print(f"Step {step}: {position}")
