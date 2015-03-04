# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from operator import itemgetter
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        # print "Best Score " , bestScore;
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = 0;

        x, y = newPos;
        ghostListPosDistance = [];
        foodListPosDistance = [];

        for ghostPos in newGhostStates:
          ghostx,ghosty = ghostPos.getPosition()
          if ghostPos.scaredTimer == 0:
            ghostListPosDistance.append(((ghostx,ghosty), manhattanDistance((x,y),(ghostx,ghosty))))
        ghostListPosDistanceSorted = sorted(ghostListPosDistance, key = itemgetter(1));

        for foodPellet in newFood.asList():
          foodx, foody = foodPellet;
          foodListPosDistance.append(((foodx,foody), manhattanDistance((x,y),(foodx,foody))))
        foodListPosDistanceSorted = sorted(foodListPosDistance, key = itemgetter(1));

        if len(foodListPosDistanceSorted) > 0:
          minFoodDistance = foodListPosDistanceSorted[0][1]; #Contains Tuple ((Foodx, Foody) distance)
        else:
          minFoodDistance = 0;

        numFoodPellets = successorGameState.getNumFood();

        for ghost in newGhostStates:
          ghostx, ghosty = ghost.getPosition();
          distance = manhattanDistance((x,y), (ghostx,ghosty))
          if ghost.scaredTimer > distance:
            score += ghost.scaredTimer - distance;

        if len(ghostListPosDistanceSorted) > 0:
          score += ghostListPosDistanceSorted[0][1];

        finalscore = score -11*numFoodPellets - minFoodDistance;
        return finalscore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def min(self, gameState, depth, index):    
      agents = gameState.getNumAgents()
      minValue= 100000000000;
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(index):
        successor = gameState.generateSuccessor(index, action)
        if index == agents - 1:
          if depth == self.depth:
            tempVal = self.evaluationFunction(successor)
          else:
            tempVal = self.max(successor, depth+1, 0)
        else:
          tempVal = self.min(successor, depth, index+1)
        tupled = (tempVal, action)
        if tempVal < minValue:
          minValue, minAction = tupled

      return minValue;


    def max(self, gameState, depth, agent):
      maxValue= -100000000;
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)

        
        tempValue = self.min(successor, depth, 1)

        tupled = (tempValue, action)
        if tempValue > maxValue:
          maxValue, bestAction = tupled;

      if depth == 1:
        return bestAction
      else:
        return maxValue
      
    def getAction(self, gameState):
        return self.max(gameState,1,0);

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestAction = self.pruney(gameState,1,0,float("-inf"), float("inf"));
        return bestAction


    def pruney(self, gameState, depth, agent, alpha, beta):
      maxValue= float("-inf");
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)
        tempValue = self.notpruney(successor, depth, 1, alpha, beta)
        tupled = (tempValue, action)
        if tempValue > beta:
          return tempValue
        if tempValue > maxValue:
          maxValue, bestAction = tupled;
        alpha = max(alpha, maxValue)
      if depth == 1:
        return bestAction
      else:
        return maxValue

    def notpruney(self, gameState, depth, index, alpha, beta):    
      agents = gameState.getNumAgents()
      minValue= float("inf");
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(index):
        successor = gameState.generateSuccessor(index, action)
        if index == agents - 1:
          if depth == self.depth:
            tempVal = self.evaluationFunction(successor)
          else:
            tempVal = self.pruney(successor, depth+1, 0, alpha, beta)
        else:
          tempVal = self.notpruney(successor, depth, index+1, alpha, beta)
        tupled = (tempVal, action)
        if tempVal < alpha:
          return tempVal;
        if tempVal < minValue:
          minValue, minAction = tupled
        beta = min(beta, minValue)
      return minValue;

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def min(self, gameState, depth, index):    
      agents = gameState.getNumAgents()
      minValue= 0;
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      numberLegalMoves = len(gameState.getLegalActions(index));
      for action in gameState.getLegalActions(index):
        p = 1 / float(numberLegalMoves);
        #print p
        successor = gameState.generateSuccessor(index, action)
        if index == agents - 1:
          if depth == self.depth:
            tempVal = self.evaluationFunction(successor)
          else:
            tempVal = self.max(successor, depth+1, 0)
        else:
          tempVal = self.min(successor, depth, index+1)
        tupled = (tempVal, action)
        
        minValue += float(p)*tempVal;

      return minValue;


    def max(self, gameState, depth, agent):
      maxValue= -100000000;
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)

        
        tempValue = self.min(successor, depth, 1)

        tupled = (tempValue, action)

        if tempValue > maxValue:
          maxValue, bestAction = tupled;

      if depth == 1:
        # print "MAXXX VALLUEE", maxValue
        # print "bestACTIONNN : ", bestAction
        return bestAction
      else:
        # print "Depth : ", depth, "Value: ", maxValue;
        return maxValue
      
    def getAction(self, gameState):
        action = self.max(gameState,1,0);
        # print "Final Move : ", action
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: In this function, we used the above Evaluation Function, 
      but instead of using the scaredTimer as a factor in determining the score,
      we used an arbitrary value, such as 10. We came about this number based on a trial and error basis.
      Also instead of setting the score to 0, we kept the running score as the originals score for every time
      the evaluation function is called. 
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    score = currentGameState.getScore()
    x, y = newPos;
    ghostListPosDistance = [];
    foodListPosDistance = [];

    for ghostPos in newGhostStates:
      ghostx,ghosty = ghostPos.getPosition()
      if ghostPos.scaredTimer == 0:
        ghostListPosDistance.append(((ghostx,ghosty), manhattanDistance((x,y),(ghostx,ghosty))))
    ghostListPosDistanceSorted = sorted(ghostListPosDistance, key = itemgetter(1));

    for foodPellet in newFood.asList():
      foodx, foody = foodPellet;
      foodListPosDistance.append(((foodx,foody), manhattanDistance((x,y),(foodx,foody))))
    foodListPosDistanceSorted = sorted(foodListPosDistance, key = itemgetter(1));

    if len(foodListPosDistanceSorted) > 0:
      minFoodDistance = foodListPosDistanceSorted[0][1]; #Contains Tuple ((Foodx, Foody) distance)
    else:
      minFoodDistance = 0;
    numFoodPellets = currentGameState.getNumFood();

    for ghost in newGhostStates:
      ghostx, ghosty = ghost.getPosition();
      distance = manhattanDistance((x,y), (ghostx,ghosty))
      if ghost.scaredTimer == 0:
        score += distance;
      elif ghost.scaredTimer > distance:
        score += 100 - distance;

    if len(ghostListPosDistance) > 0:
      score += ghostListPosDistanceSorted[0][1];

    if numFoodPellets == 0:
      return 10*currentGameState.getScore()
    if numFoodPellets <= 1:
      return currentGameState.getScore() - 2*minFoodDistance

    finalscore = score - 10*numFoodPellets - 2*minFoodDistance;

    return finalscore

# Abbreviation
better = betterEvaluationFunction

