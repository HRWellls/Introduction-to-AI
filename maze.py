from QRobot import QRobot
import random

class Robot(QRobot):
    valid_action = ['u', 'r', 'd', 'l']

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon=0.5):
        super(Robot, self).__init__(maze, alpha, gamma, epsilon)
        self.maze = maze

    def train_update(self):
        self.state = self.sense_state()  # Get the initial position of the robot
        self.create_Qtable_line(self.state)  # For the current state, retrieve the Q-table, add it to the Q-table if it does not exist          
        if random.random() < self.epsilon :
            action = random.choice(self.valid_action)
        else:
            action =max(self.q_table[self.state], key=self.q_table[self.state].get)# Choose an action
        reward = self.maze.move_robot(action)  # Move the robot with the given action (direction)
        next_state = self.sense_state()  # Get the position of the robot after performing the action
        self.create_Qtable_line(next_state)  # For the current next_state, retrieve the Q-table, add it to the Q-table if it does not exist
        self.update_Qtable(reward, action, next_state)  # Update Q-value in the Q-table
        self.update_parameter()  # Update other parameters
        return action, reward

    def test_update(self):
        self.state = self.sense_state()  # Get the initial position of the robot
        self.create_Qtable_line(self.state)  # For the current state, retrieve the Q-table, add it to the Q-table if it does not exist
        action = max(self.q_table[self.state], key=self.q_table[self.state].get)  # Choose an action
        reward = self.maze.move_robot(action)  # Move the robot with the given action (direction)
        return action, reward
