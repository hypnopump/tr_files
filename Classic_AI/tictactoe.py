# Code written entirely by Eric Alcaide
# Email: ericalcaide1@gmail.com
# GitHub: https://github.com/EricAlcaide 

# Import modules/dependencies
import random
import numpy
from copy import deepcopy

# Create TicTacToe Class (OOP)
class TicTacToe():
	dims = 3
	ends = [-1, 0, 1] 			# -1 if User wins, 0 if Draw, +1 if Computer wins
	exchange = {-1: "You Win! Coungratulations!", 0: "Draw!", 1: "I win... sorry"}
	grid = []
	# Print the instructions
	def instructions(self):
		print("Hi! I'm a program created by Eric Alcaide with the purpose of never losing at Tic Tac Toe.")
		print("First, I'll give you some instructions:")
		print("The grid we are going to use looks like this:")
		self.showGrid()
		print("You will be an 'X' and I'll be an 'O'. Your goal is... well, I supose you already know it!")
		print("The cells are labeled with two numbers, corresponding to cols and rows.")
		print("For example, 1.1 will be the top-left corner, 3.1 will be the bottom-left one,")
		print("1.3 will be the top-right corner and 3.3 will be the bottom-right one.")
		print("Enough! You're ready to play!")
		print("I'll let you move first...")
		print()

	# Create the grid
	def createGrid(self):
		for i in range(self.dims):
			row = [ " " for j in range(self.dims) ]
			self.grid.append(row)

	# Display the grid
	def showGrid(self):
		print("-------------")
		for i in self.grid:
			line = "|"
			for j in i:
				line = line+" "+str(j)+" |"
			print(line)
			print("-------------")

	# Returns +1 if Computer wins, -1 if User wins, 0 if Draw
	def whoWins(self, grid):
		# Check horizontal matches
		for i in range(3):
			if grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0] != " ":
				return -1 if grid[i][1] == "X" else 1
		# Check vertical matches
		for i in range(3):
			if grid[0][i] == grid[1][i] == grid[2][i] and grid[0][i] != " ":
				return -1 if grid[1][i] == "X" else 1
		# Check Diagonals
		if (grid[0][0] == grid[1][1] == grid[2][2] or grid[2][0] == grid[1][1] == grid[0][2]) and grid[1][1] != " ":
			return -1 if grid[1][1] == "X" else 1
		# Draw!
		if len(self.movements(grid)) == 0:
			return 0
		# No winner yet
		return 5

	# Returns possible movements
	def movements(self, grid):
		children = []
		for i in range(3):
			for j in range(3):
				if grid[i][j] == " ":
					children.append((i,j))
		return children

	# Minimax Algorithm with Alpha Beta - Max tries to maximize the result while Min tries to minimize it
	def maximize(self, grid, alpha, beta):
		# Ensure it's not a terminal state
		if self.whoWins(grid) != 5:
			# print("winner", self.whoWins(grid))
			return "nothing here max", self.whoWins(grid)
		# Expand node
		max_utility = -100
		children = self.movements(grid)
		# Evaluate children
		for child in children:
			test_grid = deepcopy(grid)
			test_grid[child[0]][child[1]] = "O"
			utility = self.minimize(test_grid, alpha, beta)[1]
			if utility > max_utility:
				(max_child, max_utility) = (test_grid, utility)
			if max_utility >= beta:
				break
			if max_utility > alpha:
				alpha = max_utility
		return max_child, max_utility

	def minimize(self, grid, alpha, beta):
		# Ensure it's not a terminal state
		if self.whoWins(grid) != 5:
			return "nothing here", self.whoWins(grid)
		# Expand node
		min_utility = 100
		children = self.movements(grid)
		# Evaluate children
		for child in children:
			test_grid = deepcopy(grid)
			test_grid[child[0]][child[1]] = "X"
			utility = self.maximize(test_grid, alpha, beta)[1]
			if utility < min_utility:
				(min_child, min_utility) = (test_grid, utility)
			if min_utility <= alpha:
				break
			if min_utility < beta:
				beta = min_utility
		return min_child, min_utility

	def decision(self, grid): 	# Modifies the grid w/ the decision made by Max
		(alpha,beta) = (-100,+100)
		print()
		print("My turn: ")
		print()
		self.grid = self.maximize(grid, alpha, beta)[0]
		return

	# Request the user for his/her move
	def requestMove(self):
		print()
		print("Your turn:")
		move = str(input())
		if move == "stop":
			print(99+"hello"+['hi', 9])
		if len(move) == 3:
			(x,y) = (int(move[0])-1, int(move[2])-1)
			if x >= 0 and x <= 2 and y >= 0 and y <= 2:
				if self.grid[x][y] == " ":
					self.grid[x][y] = "X"
				else:
					print("Invalid Movement")
					self.requestMove()
			else:
				print("Invalid Movement")
				self.requestMove()
		else:
			print("Invalid Movement")
			self.requestMove()
		print()

def main():
	tictactoe = TicTacToe()
	tictactoe.createGrid()
	tictactoe.instructions()
	tictactoe.showGrid()

	while True:
		# User moves
		tictactoe.requestMove()
		tictactoe.showGrid()
		if tictactoe.whoWins(tictactoe.grid) != 5:
			print(tictactoe.grid)
			print(tictactoe.whoWins(tictactoe.grid))
			print(tictactoe.exchange[tictactoe.whoWins(tictactoe.grid)])
			break
		# Computer moves
		tictactoe.decision(tictactoe.grid)
		tictactoe.showGrid()
		if tictactoe.whoWins(tictactoe.grid) != 5:
			print()
			print(tictactoe.exchange[tictactoe.whoWins(tictactoe.grid)])
			break

# Run the program
if __name__ == "__main__":
	main()