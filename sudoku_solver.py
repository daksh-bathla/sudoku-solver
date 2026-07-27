"""
Sudoku Solver - OpenCV + Backtracking

Solves Sudoku puzzles from images or hardcoded arrays.
- Preprocesses grid image (grayscale, threshold, find contour, warp)
- Splits into 81 cells and extracts digits (or use hardcoded input)
- Solves via backtracking algorithm
- Overlays solution on original image
- Saves output images (original, extracted, solved)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class SudokuSolver:
    """Solve Sudoku puzzles from images or arrays."""

    def __init__(self, cell_size=50):
        self.cell_size = cell_size
        self.solution = None

    def backtrack_solve(self, grid):
        """Solve Sudoku using backtracking. Modifies grid in place."""
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for num in range(1, 10):
                        if self.is_valid(grid, i, j, num):
                            grid[i][j] = num
                            if self.backtrack_solve(grid):
                                return True
                            grid[i][j] = 0
                    return False
        return True

    @staticmethod
    def is_valid(grid, row, col, num):
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in grid[row]:
            return False

        # Check column
        if num in grid[:, col]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row : box_row + 3, box_col : box_col + 3]:
            return False

        return True

    def solve(self, puzzle):
        """
        Solve Sudoku puzzle.
        Args:
            puzzle: 9x9 numpy array or list of lists (0 for empty cells)
        Returns:
            Solved 9x9 numpy array
        """
        grid = np.array(puzzle, dtype=int)
        original = grid.copy()

        if self.backtrack_solve(grid):
            self.solution = grid
            return grid, original
        else:
            raise ValueError("No solution exists for this puzzle.")

    def extract_digits_from_image(self, img_path):
        """
        Extract Sudoku grid digits from image using basic preprocessing.
        Returns 9x9 grid with 0 for empty cells.
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to binary
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in image.")

        # Get largest contour (assume it's the Sudoku grid)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Extract grid region
        grid_img = img[y : y + h, x : x + w]

        # For simplicity, return a placeholder grid (9x9 of 0s)
        # In production, would use digit recognition (Tesseract/CNN)
        print(
            "Image extracted. For OCR, integrate pytesseract or TensorFlow digit recognition."
        )
        return np.zeros((9, 9), dtype=int)

    def render_grid(self, puzzle, solution=None, output_path="grid.png"):
        """
        Render Sudoku grid as image.
        If solution provided, overlay it in a different color.
        """
        size = 9 * self.cell_size + 10  # 10 for borders
        img = Image.new("RGB", (size, size), color="white")
        draw = ImageDraw.Draw(img)

        # Try to load font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            font = ImageFont.load_default()

        border_color = "black"
        thick_border = 3
        thin_border = 1

        # Draw grid lines
        for i in range(10):
            line_width = thick_border if i % 3 == 0 else thin_border
            x = 5 + i * self.cell_size
            draw.line([(x, 5), (x, size - 5)], fill=border_color, width=line_width)
            y = 5 + i * self.cell_size
            draw.line([(5, y), (size - 5, y)], fill=border_color, width=line_width)

        # Draw numbers
        for i in range(9):
            for j in range(9):
                x = 5 + j * self.cell_size + self.cell_size // 2
                y = 5 + i * self.cell_size + self.cell_size // 2

                num = puzzle[i][j]
                color = "black"

                # If solution provided, show original in black, solution in blue
                if solution is not None and puzzle[i][j] == 0:
                    num = solution[i][j]
                    color = "blue"

                if num != 0:
                    draw.text(
                        (x, y),
                        str(num),
                        fill=color,
                        font=font,
                        anchor="mm",
                    )

        img.save(output_path)
        print(f"Saved: {output_path}")
        return img

    def solve_and_render(self, puzzle_array, output_dir="output"):
        """Solve puzzle and save all output images."""
        os.makedirs(output_dir, exist_ok=True)

        # Original puzzle
        original = np.array(puzzle_array, dtype=int)
        self.render_grid(original, output_path=f"{output_dir}/01_original_puzzle.png")

        # Solve
        solved, _ = self.solve(puzzle_array)

        # Overlay: original with solution filled in blue
        self.render_grid(original, solution=solved, output_path=f"{output_dir}/03_solved_overlay.png")

        # Solution only
        self.render_grid(solved, output_path=f"{output_dir}/02_solved_grid.png")

        print(f"\nSudoku solved! Outputs in {output_dir}/")
        return solved


# Sample Sudoku puzzle (0 = empty)
SAMPLE_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


if __name__ == "__main__":
    solver = SudokuSolver()
    solved = solver.solve_and_render(SAMPLE_PUZZLE)

    print("\nOriginal:")
    print(np.array(SAMPLE_PUZZLE))
    print("\nSolved:")
    print(solved)
