from copy import deepcopy
from random import choice
from math import sqrt
from math import log
from sys import maxsize
from game import Game

 
class Node:
    def __init__(self, Boarding, Parent=None, action=None, color=""):
        self.Visit = 0  
        self.reward = 0.0 
        self.Boarding = Boarding  
        self.Children = []  
        self.Parent = Parent 
        self.action = action 
        self.color = color  
 
    def ucb(self):#calulate the ucb of this node
        if self.Visit == 0:#never visited -> visit first
            return maxsize  
 
        past=self.reward/self.Visit#expectation
        future = sqrt(2.0 * log(self.Parent.Visit) / float(self.Visit))#potential
        ucb_of_node = past + future#ucb
        return ucb_of_node
 

    def addchild(self, child_Boarding, action, color):#add the child after this action
        kid = Node(child_Boarding, Parent=self, action=action, color=color)
        self.Children.append(kid)
 
  
    def examine(self):#examine if there is still some children which have never been visited.If is,then visit if first;
        if len(self.Children) == 0:
            return False
        for kid in self.Children:
            if kid.Visit == 0:
                return False
        
        return True
 
 
class AIPlayer:

 
    def __init__(self, color):
 
        self.color = color
 
    def MCM(self, root):
       
 
        for i in range(100):  
            selected_node = self.select(root)#find the leafnode with 0 visit.
            leaf_node = self.expand(selected_node)#if every leaf is visited,then creat a new leaf.
            reward = self.stimulate(leaf_node)#simple stimulate
            self.back(leaf_node, reward)#feedback
 
        max_node = None    
        max_ucb = -maxsize
        for child in root.Children:
            child_ucb = child.ucb()
            if max_ucb < child_ucb:
                max_ucb = child_ucb
                max_node = child  
 
        return max_node.action
 
    def select(self, node):
        if len(node.Children) == 0:  
            return node#If it's the leaf , stop the recursion #situation 1
        if node.examine():    #It means it doesn't have a child which hasn't ever been visited,resulting in high potential.
            max_node = None
            max_ucb = -maxsize
            for child in node.Children:#visit the childnode with highest ucb
                child_ucb = child.ucb()
                if max_ucb < child_ucb:
                    max_ucb = child_ucb
                    max_node = child   
            return self.select(max_node)
 
        else:  
            for kid in node.Children:  
                if kid.Visit == 0:
                    return kid#Visit the childnode (the never visited one) with superhigh potential.#situation 2
 
    def expand(self, node):
        if node.Visit == 0:#situation 2 in "leaf" program    
            return node#no need to expand
        #do the expansion   
        if node.color == 'X':
            new_color = 'O'
        else:
            new_color = 'X'
        for action in list(node.Boarding.get_legal_actions(node.color)):  #expand all potential locations of the leaf.
            new_board = deepcopy(node.Boarding)
            new_board._move(action, node.color)
            node.addchild(new_board, action=action, color=new_color)
        if len(node.Children) == 0:# the node can't be expanded!
            return node
        return node.Children[0]    
 
    def stimulate(self, node): 
        board = deepcopy(node.Boarding)
        color = node.color
        count = 0
        while (not self.gamedone(board)) and count < 100:   
            action_list = list(node.Boarding.get_legal_actions(color))
            if not len(action_list) == 0:   
                action = choice(action_list)
                board._move(action, color)
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
            else:   #no actions follow -> change the player
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
                action_list = list(node.Boarding.get_legal_actions(color))
                action = choice(action_list)
                board._move(action, color)
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
            count = count + 1
 
        winner, difference = board.get_winner()
        if winner == 2:
            reward = 0
        elif winner == 0:    
            reward = 20 + difference
        else:
            reward = -(20 + difference)
 
        if self.color == 'X':
            reward = - reward
 
        return reward
 
    def back(self, node, reward):

        while node is not None:
            node.Visit += 1
            if node.color == self.color:
                node.reward += reward
            else:
                node.reward -= reward
            node = node.Parent
        return 0
 
    def gamedone(self, board):
        if (len(list(board.get_legal_actions('X'))) == 0 and len(list(board.get_legal_actions('O'))) == 0)  :
            return True
        return False
 
    def get_move(self, board):
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
 
        
        root = Node(Boarding=deepcopy(board), color=self.color)
 
        action = self.MCM(root)
       
 
        return action
 
 

