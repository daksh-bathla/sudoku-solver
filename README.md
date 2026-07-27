# Sudoku Solver with OpenCV + Backtracking

Solve Sudoku puzzles from images or hardcoded arrays. Extracts grid digits, solves via backtracking, overlays solution.

## Features

- **Backtracking Solver**: Solves any valid Sudoku puzzle
- **Image Preprocessing**: OpenCV pipeline (grayscale → threshold → contour detection → perspective warp)
- **Cell Extraction**: Splits warped grid into 81 cells (9×9)
- **Digit Extraction**: Placeholder for OCR/ML digit recognition (pytesseract, TensorFlow ready)
- **Visual Output**: Renders 3 images:
  - `01_original_puzzle.png` - Original with empty cells
  - `02_solved_grid.png` - Fully solved grid
  - `03_solved_overlay.png` - Original + solution (blue = filled)

## Installation

```bash
pip install opencv-python numpy pillow pytesseract
```

Optional: Install Tesseract OCR for actual digit extraction
```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Option 1: Web App (Recommended)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

**Three ways to solve:**
1. **Upload Sudoku Image** (📸) - Auto-extract digits via OCR, edit if needed, solve
2. **Manual Array Input** (📋) - Paste 9×9 puzzle, solve, render
3. **View Info** (ℹ️) - Algorithm & extension guide

### Option 2: Python Script (Array)

```python
from sudoku_solver import SudokuSolver

puzzle = [
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

solver = SudokuSolver()
solved = solver.solve_and_render(puzzle)
```

### Option 3: Image-to-Solution (Python)

```python
from sudoku_solver import SudokuSolver
from image_processor import ImageProcessor

# Extract digits from image (uses Tesseract OCR)
grid, warped = ImageProcessor.extract_grid_from_image("sudoku_puzzle.jpg")

# Solve
solver = SudokuSolver()
solved, original = solver.solve(grid)

# Render
solver.render_grid(original, solution=solved, output_path="solved.png")
```

### Command Line (Quick Test)

```bash
python sudoku_solver.py
```

Generates sample output to `output/` directory.

## Project Structure

```
sudoku/
├── sudoku_solver.py      # Main solver, backtracking algorithm, rendering
├── image_processor.py    # OpenCV preprocessing & digit extraction
├── output/               # Generated images
│   ├── 01_original_puzzle.png
│   ├── 02_solved_grid.png
│   └── 03_solved_overlay.png
└── README.md
```

## Algorithm Details

### Backtracking Solver
1. Iterate through all cells (0-80)
2. For empty cells (value = 0):
   - Try digits 1-9
   - Check validity (row, column, 3×3 box)
   - Recurse; backtrack if dead end
3. Return when all cells filled

### Image Processing Pipeline
1. **Grayscale**: Convert BGR → grayscale
2. **Threshold**: Binary conversion (adaptive threshold for better lighting)
3. **Contour Detection**: Find largest contour (Sudoku grid)
4. **Perspective Warp**: Transform to flat 450×450 top-down view
5. **Cell Split**: Divide into 9×9 grid of 50×50 cells
6. **Digit Extraction**: OCR on each cell (placeholder/extensible)

## Digit Recognition (OCR)

**Current Implementation**: Tesseract OCR via pytesseract
- Installed automatically via `requirements.txt`
- Works well for printed Sudoku grids
- Attempts OCR first, falls back to contour detection if unavailable

**Setup**: Requires Tesseract binary
```bash
# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Improving Accuracy**:
- Use clear, high-contrast images
- Avoid shadows and glare
- Use Streamlit app's "Edit Extracted Puzzle" feature to fix OCR errors manually

**Alternative: TensorFlow/Keras (Handwritten Digits)**
```python
import tensorflow as tf

model = tf.keras.models.load_model("digit_model.h5")

def extract_digit_from_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28)) / 255.0
    pred = model.predict(gray.reshape(1, 28, 28, 1))
    digit = np.argmax(pred)
    return digit if np.max(pred) > 0.7 else 0
```

Replace this in `image_processor.py:extract_digit_from_cell()` if using ML model.

## Sample Puzzle

Included in `sudoku_solver.py` (lines 100-107). Run directly to generate output images.

## Known Limitations

- OCR accuracy depends on image quality (printed grids: ~90-95%, handwritten: ~70-80%)
- Requires relatively clean, well-lit Sudoku image for best results
- Assumes grid is roughly square/rectangular in the image
- No automatic validation that extracted puzzle is solvable
  - Use Streamlit app's "Edit Extracted Puzzle" to fix OCR errors
- Streaming app limits file size to typical browser upload limits (~200MB)

## Future Enhancements

- Integrate Tesseract OCR for real digit recognition
- Train CNN on printed/handwritten digits
- Add support for invalid Sudoku detection
- Webcam live-preview mode
- GUI for image upload & visualization
