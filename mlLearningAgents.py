# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np

# Machine Learning CW2
# Steven Vuong / 1871066
# 24/03/19

# class to handle food matrix (of T and F values)
class Food(object):
    def __init__(self, food_matrix):
        '''Initialise food matrix'''
        self.food_matrix = food_matrix
        self.food_list = None

    # get food coordinates from matrix
    def getFoodCoordsFromMatrix(self):
        food_matrix = self.food_matrix
        food_coords = []
        
        # loop through food matrix and return coordinates
        for i in range(len(food_matrix[0])):
            line = food_matrix[i]
            for j in range(len(line)):
                boolean = line[j]
                if boolean == True:
                    food_coords.append((i,j))

        self.food_list = food_coords
        return food_coords

    # return manhattan distance between coordinates (more general method)
    def manhattanDistance(self, start, end):
        return (abs(end[1]-start[1]) + abs(end[0]-start[0]))

    # get closest food
    def getClosestFood(self, position):
        food_array = self.food_list
        # initialise parameters
        min_distance = 30
        closest_food = None

        # loop through foods
        for food in food_array:
            distance = self.manhattanDistance(food, position)
            if distance<min_distance:
                closest_food = food
                min_distance = distance

        return closest_food


# class to hold object containing positions of agents
class AgentPosition(object):
    def __init__(self, agent_positions):
        '''
        Initialise state object which holds the following:
        @param agent_positions = (pacman position, ghosts positions)
        '''
        self.agent_positions = agent_positions
        self.pacman_position = agent_positions[0]
        # could be multiple ghosts, stores as list
        self.ghost_positions = agent_positions[1]

    # standard getter functions for agent positions
    def getPacmanPos(self):
        return self.pacman_position

    def getGhostPos(self):
        return self.ghost_positions

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 2000):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # initialise storage for state objects, dict of {(pacman, ghosts):{action: q-value}}
        self.states_dict = {}
        # initialise variables to hold information about previous move
        self.lastPosition = None
        self.lastScore = None
        # counter variable
        self.moveCounter = 0

    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    ## Accessor functions for agents' storage (self.states_dict{})

    # returns highest action for action:q-value dict
    def getMaxActionInner(self, action_value_dict):
        q_values = list(action_value_dict.values())
        actions = list(action_value_dict.keys())
        return actions[q_values.index(max(q_values))]

    # returns highest q-value for action:q-value dict
    # if empty list, return 0
    def getMaxQInner(self, action_value_dict):
        q_values = list(action_value_dict.values())
        if len(q_values) == 0:
            return 0
        return max(q_values)

    # return legal action with highest q-value for given agent position input
    # if no best action, return none
    def getBestAction(self, agent_positions, legal):

        # initialise best action
        best_action = None
        
        # get best action from action-value dict
        if agent_positions in self.states_dict.keys():

            action_value_dict = self.states_dict.get(agent_positions)
            best_action = self.getMaxActionInner(action_value_dict)

        # if key doesn't exist, create action q-value pairs with q=0
        else:
            action_value_dict = {}
            for action in legal:
                action_value_dict[action] = 0

            self.states_dict[agent_positions] = action_value_dict
            
        return best_action

    # get {action:qvalue} dict for agent_position inputs
    # if doesn't exist, return 0 and create pairing
    def getQ(self, agent_positions, action):

        # initialise q-value
        q_value = 0

        # if dict exists, get action-qvalue dict
        if agent_positions in self.states_dict.keys():
            action_q_dict = self.states_dict.get(agent_positions)
            
            # if action exists there, get q-value 
            if action in action_q_dict.keys():
                q_value = action_q_dict.get(action)
            
            # otherwise set q-value to 0 for action and update
            else:
                action_q_dict[action] = 0

                self.states_dict[agent_positions] = action_q_dict

        # if dict doesn't exist, create and input instance for action
        # setting q to 0
        else:
            action_q_dict = {}
            action_q_dict[action] = 0

            self.states_dict[agent_positions] = action_q_dict

        return q_value

    # get best q value for any given state
    # if doesn't exist, return 0 and update
    def getMaxQOuter(self, agent_positions, legal):

        # initialise max q to 0
        max_q = 0

        # see if the key exists in dict, if so get action value pairs
        if agent_positions in self.states_dict.keys():
            action_q_dict = self.states_dict.get(agent_positions)

            # then select highest q value
            max_q = self.getMaxQInner(action_q_dict)

        # otherwise create dict and append action:q-value pairs, q-value=0
        else:
            action_value_dict = {}
            for action in legal:
                action_value_dict[action] = 0

        return max_q

    # update q-value
    def updateQ(self, last_agentPosition, last_action, max_q, reward):
            last_q = self.getQ(last_agentPosition, last_action)

            # calculate updated q value
            delta_q = reward + (self.gamma*max_q) - last_q
            new_q = last_q + self.alpha*delta_q

            # need to update the dict and action pairing
            action_value_dict = self.states_dict.get(last_agentPosition)

            # then update dict and pass on to update the state action pair
            action_value_dict[last_action] = new_q

            # this one
            self.states_dict[last_agentPosition] = action_value_dict
    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        last_action = state.getPacmanState().configuration.direction
        pacman_position = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()
        score = state.getScore()

        # find information about food
        food_matrix = state.getFood()
        foodObject = Food(food_matrix)
        food_list = foodObject.getFoodCoordsFromMatrix()
        closest_food = foodObject.getClosestFood(pacman_position)

        # compute best action to take, use epsilon to explore, otherwise random
        agent_positions_tuple = (pacman_position, ghost_positions)

        agentPosition = None
        # if agentPosition exists as a key already then use that
        not_in = True
        for agent_positions in self.states_dict.keys():
            
            agent_pman_pos = agent_positions.getPacmanPos()
            agent_ghosts_pos = agent_positions.getGhostPos()

            if ((pacman_position == agent_pman_pos) and (ghost_positions == agent_ghosts_pos)):
                agentPosition = agent_positions
                not_in = False

        if not_in == True:
            agentPosition = AgentPosition(agent_positions_tuple)

        # get best-action pair for agent_positions
        best_action = self.getBestAction(agentPosition, legal)

        # update q after the first move (first move is 'Stop')
        if self.moveCounter > 0:
            # update q
            # get previous co-ordinate given current co-ordinate and last action
            last_agentPosition = self.lastPosition
            last_score = self.lastScore
            # get max q-value in current co-ordinate for all actions
            max_q = self.getMaxQOuter(agentPosition, legal)
            # calculate reward
            reward = score-last_score
            # update previous co-ordinate action pairing
            self.updateQ(last_agentPosition, last_action, max_q, reward)
            
        # If no best action, pick random
        # or if random value is less than epsilon
        rand = np.random.uniform(0, 1)
        if ((best_action is None) or (rand < self.epsilon)):
            best_action = random.choice(legal)

        # record (and update) last position and score
        self.lastPosition = agentPosition
        self.lastScore = score
        self.moveCounter += 1

        # We have to return an action
        return best_action
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # get information about state it was just in before dying
        last_position = self.lastPosition
        last_score = self.lastScore
        last_action = state.getPacmanState().configuration.direction

        # get information about current states (in death)
        pacman_position = state.getPacmanPosition() 
        score = state.getScore()

        # max_q is zero anyway, since legal is empty
        legal = state.getLegalPacmanActions()
        max_q = self.getMaxQOuter(pacman_position, legal)

        # calculate reward
        reward = score - self.lastScore
        
        # update previous co-ordinate action pairing
        self.updateQ(last_position, last_action, max_q, reward)
        self.moveCounter = 0

        # print "A game just ended!"

        # Something to note: Interestingly pacman performed worse when we
        # change the learning rate and epsilon over times so it was decided
        # to keep alpha and epsilon fixed
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


