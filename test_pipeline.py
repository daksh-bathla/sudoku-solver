"""Test the full image → extraction → solve pipeline."""

import numpy as np
from image_processor import ImageProcessor
from sudoku_solver import SudokuSolver

print("Testing full pipeline: image → extract → solve")
print("=" * 50)

# Extract grid from test image
try:
    print("1️⃣ Extracting grid from test_sudoku.jpg...")
    grid, warped = ImageProcessor.extract_grid_from_image("test_sudoku.jpg")
    print(f"✅ Extracted grid shape: {grid.shape}")
    print("Extracted puzzle:")
    print(grid)
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    exit(1)

# Solve
try:
    print("\n2️⃣ Solving puzzle...")
    solver = SudokuSolver()
    solved, original = solver.solve(grid)
    print("✅ Solved!")
    print("Solution:")
    print(solved)
except Exception as e:
    print(f"❌ Solving failed: {e}")
    exit(1)

# Render
try:
    print("\n3️⃣ Rendering images...")
    solver.render_grid(original, output_path="test_original.png")
    solver.render_grid(original, solution=solved, output_path="test_solved.png")
    print("✅ Rendered test_original.png and test_solved.png")
except Exception as e:
    print(f"❌ Rendering failed: {e}")
    exit(1)

print("\n" + "=" * 50)
print("✅ Pipeline test complete!")
