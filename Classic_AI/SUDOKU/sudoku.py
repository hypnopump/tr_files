# Sudoku solver AI
# Author: Eric Alcaide
# Email: ericalcaide1@gmail.com

# Importing libraries
import sys

class Sudoku():
	sudokuBoard = {}
	sudokuList = []
	sudokuRows = "ABCDEFGHI"
	sudokuCols = "123456789"
	sudokuSquares = (
		("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"),
		("D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3"),
		("G1", "G2", "G3", "H1", "H2", "H3", "I1", "I2", "I3"),

		("A4", "A5", "A6", "B4", "B5", "B6", "C4", "C5", "C6"),
		("D5", "D5", "D6", "E4", "E5", "E6", "F4", "F5", "F6"),
		("G5", "G5", "G6", "H4", "H5", "H6", "I4", "I5", "I6"),

		("A7", "A8", "A9", "B7", "B8", "B9", "C7", "C8", "C9"),
		("D7", "D8", "D9", "E7", "E8", "E9", "F7", "F8", "F9"),
		("G7", "G8", "G9", "H7", "H8", "H9", "I7", "I8", "I9")
	)
	peers = dict((s, self.neighborhood(s)) for s in sudokuList)
	arcs = []
	steps = []

	def __init__(self, string):
		# Create the list with cells and the sudokuBoard
		self.sudokuList = [row+col for col in self.sudokuCols for row in self.sudokuRows]
		for cell in self.sudokuList: self.sudokuBoard[cell] = 0 
		# Create the list of arcs
		# Rows and cols here
		for cell in self.sudokuList:
			row, col = cell[0], cell[1]
			for related in self.sudokuBoard:
				if related != cell:
					if row in related or col in related:
						self.arcs.append((cell, related))
		# Squares
		for square in self.sudokuSquares:
			for cell in square:
				for related in square:
					if cell != related and (cell, related) not in self.arcs:
						self.arcs.append((cell, related))
		# Fill the sudokuBoard with values of the string
		for cell,value in zip(self.sudokuList, string):
			if value != "0" and value != ".":
				self.sudokuBoard[cell] = value
			else:
				self.sudokuBoard[cell] = "123456789"


	def neighborhood(self, pos):
		""" Returns all cells affected by a given cell. """
		neighborhood = []
		for pair in self.arcs:
			if (pair[1] == pos):
				neighborhood.append(pair[0])
		return neighborhood

	def neighbors(self, pos, x_prev):
		""" Returns all relations between cells as binary constraints. """
		neighbors = []
		if x_prev != False:
			for pair in self.arcs:
				if (pair[1] == pos) and (pair[0] != x_prev):
					neighbors.append(pair)
		else:
			for pair in self.arcs:
				if (pair[1] == pos):
					neighbors.append(pair)
		return neighbors

	def AC3(self, board):
		""" Runs AC3 Constraint propagation algorithm. """
		arcs = self.arcs[:]
		while len(arcs) > 0:
			pair = arcs.pop()
			# Declaring some variables
			pair_0 = pair[0]
			pair_1 = pair[1]
			domain_0 = board[pair_0]
			domain_1 = board[pair_1]
			if self.revise(pair_0, pair_1, domain_0, domain_1, board) == True:
				if len(domain_0) == 0:
					return False
				arcs.extend(self.neighbors(pair_0, pair_1))
			
		return board

	def revise(self, pair_0, pair_1, domain_0, domain_1, board):
		""" Revises if constraint is satisfiable """
		revised = False
		for x in domain_0:
			aux = True
			for y in domain_1:
				if y != x: aux = True; break # Constraint satisfied
				else: aux = False

			if aux == False:
				board[pair_0] = board[pair_0].replace(x, '')
				revised = True
		return revised

	def bactracking_search(self):
		"""Backtracking search algorithm manager"""
		final = self.backtrack(self.sudokuBoard)
		if final != False:
			self.sudokuBoard = final
			self.grid(final)
		return

	def backtrack(self, board):
		"""Backtracking search algorithm"""
		if board is False:
			return False # Saving time
		if all(len(board[cell]) == 1 for cell in self.sudokuList): 
			return board # Solved!
		try: 
			length,cell = min((len(board[cell]), cell)
								for cell in self.sudokuList
								if len(board[cell]) > 1)
		except(ValueError):
			return board

		for value in board[cell]:
			changed = self.AC3(self.changed(board.copy(), cell, value))
			if self.checked(changed) == True:
				# Recording step
				self.steps.append(changed)
				result = self.backtrack(changed)
				if result != False:
					return result
		
		return False

	def select_unassigned_variables(self, board):
		""" Return all cells with >1 possible values. The ones with
			less options are given first."""
		unassigned = [cell for cell in self.sudokuList
						if len(self.sudokuBoard[cell]) > 1]

		return sorted(unassigned, key=lambda x: len(x))

	def changed(self, board, cell, value):
		""" Change a cell value. """
		board[cell] = value
		return board

	def checked(self, board):
		""" Returns True if there's no conflict, False otherwise. """
		for cell in self.sudokuList:
			for neighbor in self.neighborhood(cell):
				if len(board[cell]) == 1:
					if board[neighbor] == board[cell]  or len(board[neighbor]) == 0:
						return False
		return True

	def grid(self, board):
		"""Display grid in human-readable format."""
		aux = 0
		line = " "
		sep_lines = "--+----------+----------+----------+--"
		for cell in self.sudokuList:
			value = board[cell]
			if aux%27 == 0:
				print(sep_lines)
			if aux%3 == 0:
				line = line+" "+"|"
			line = line+"  "+str(value)
			if aux%9 == 8:
				print(line+" "+"|")
				line = " "
			aux = aux+1
		print(sep_lines)
		return

def main(input_text):
	""" Just Do It. """
	sudoku 		= Sudoku(input_text)
	sudoku.steps.append(sudoku.sudokuBoard)
	sudoku.sudokuBoard = sudoku.AC3(sudoku.sudokuBoard)
	sudoku.bactracking_search()

	try:
		from visualize import visualize_assignments
		visualize_assignments(sudoku.steps)
	except:
		print("Visualizing tool is so badass. Sorry but it doesn't work correctly.")
	
if __name__ == '__main__':
	input_text = "040000008000006000207840603708100260000700030009030000050000000000300125300021000"
	main(input_text)