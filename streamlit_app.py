"""
Sudoku Solver Streamlit App

Interactive web interface for solving Sudoku puzzles from images or arrays.
"""

import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import tempfile
import os

from sudoku_solver import SudokuSolver
from image_processor import ImageProcessor


st.set_page_config(page_title="Sudoku Solver", layout="wide")

st.title("🧩 Sudoku Solver")
st.write(
    "Upload a Sudoku puzzle image → Get solved puzzle image. Or enter puzzle manually."
)

# Sample puzzle for reference
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

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["🖼️ Image Upload", "📋 Array Input", "ℹ️ Info"])

with tab1:
    st.subheader("Upload Sudoku Image")
    st.write("📸 Upload a photo/scan of a Sudoku puzzle. We'll extract, solve, and show you the answer.")

    uploaded_file = st.file_uploader("Choose image...", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Show uploaded image
            uploaded_img = Image.open(uploaded_file)

            col_info, col_img = st.columns([1, 2])
            with col_info:
                st.write("**Uploaded Image**")
            with col_img:
                st.image(uploaded_img, use_container_width=True)

            # Extract grid with progress
            with st.spinner("🔍 Extracting Sudoku grid..."):
                gray, thresh, original_img = ImageProcessor.preprocess(tmp_path)
                grid, warped = ImageProcessor.extract_grid_from_image(tmp_path)

            st.success("✅ Grid extracted!")

            # Show extracted grid array
            with st.expander("📊 Show Extracted Numbers (click to expand)"):
                st.write(grid)
                st.write("(0 = empty cell)")
                st.info("💡 If OCR missed digits or got them wrong, edit below and re-solve.")

            # Option to edit extracted grid
            with st.expander("✏️ Edit Extracted Puzzle (if needed)"):
                st.write("Correct any errors in the extracted puzzle before solving:")
                edited_text = st.text_area(
                    "Edit grid:",
                    value="\n".join([" ".join(map(str, row)) for row in grid]),
                    height=150,
                    key="edited_grid"
                )

                try:
                    lines = edited_text.strip().split("\n")
                    edited_grid = []
                    for line in lines:
                        row = [int(x) for x in line.replace(",", " ").split() if x]
                        if len(row) == 9:
                            edited_grid.append(row)
                    if len(edited_grid) == 9:
                        grid = np.array(edited_grid)
                        st.success("✅ Grid updated!")
                except:
                    st.warning("⚠️ Invalid format. Keep original grid.")

            # Solve
            with st.spinner("🧮 Solving puzzle..."):
                solver = SudokuSolver()
                try:
                    solved, original = solver.solve(grid)
                    st.success("✅ Puzzle solved!")

                    # Display side-by-side
                    sol_col1, sol_col2 = st.columns(2)

                    with sol_col1:
                        st.subheader("Original Puzzle")
                        img_original = solver.render_grid(original)
                        st.image(img_original, use_container_width=True)

                    with sol_col2:
                        st.subheader("✅ Solution (Blue = Filled)")
                        img_solved = solver.render_grid(original, solution=solved)
                        st.image(img_solved, use_container_width=True)

                    # Download buttons
                    st.subheader("📥 Download Results")
                    dl_col1, dl_col2, dl_col3 = st.columns(3)

                    buf1 = BytesIO()
                    img_original.save(buf1, format="PNG")
                    dl_col1.download_button(
                        "Original Puzzle",
                        buf1.getvalue(),
                        "sudoku_original.png",
                        "image/png",
                    )

                    buf2 = BytesIO()
                    img_solved.save(buf2, format="PNG")
                    dl_col2.download_button(
                        "Solved (with answer)",
                        buf2.getvalue(),
                        "sudoku_solved.png",
                        "image/png",
                    )

                    solution_text = "\n".join([" ".join(map(str, row)) for row in solved])
                    dl_col3.download_button(
                        "Numbers Only",
                        solution_text,
                        "sudoku_solution.txt",
                        "text/plain",
                    )

                except ValueError as e:
                    st.error(f"❌ Cannot solve: {e}")
                    st.info("💡 Try uploading a clearer image or check if the puzzle is valid.")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            st.info("💡 Make sure the image is a clear photo of a Sudoku grid.")

        finally:
            os.unlink(tmp_path)
    else:
        st.info("👆 Upload an image to get started. Try a clear photo from your phone or a scanned page.")

with tab2:
    st.subheader("Enter Sudoku Puzzle Manually")
    st.write("Use 0 for empty cells. Rows separated by newlines, values by spaces or commas.")

    # Text area for puzzle input
    puzzle_text = st.text_area(
        "Puzzle (9 rows, 9 values each):",
        value="\n".join([" ".join(map(str, row)) for row in SAMPLE_PUZZLE]),
        height=200,
    )

    col1, col2 = st.columns(2)

    if col1.button("🔍 Solve", key="solve_array"):
        try:
            # Parse input
            lines = puzzle_text.strip().split("\n")
            puzzle = []
            for line in lines:
                row = [int(x) for x in line.replace(",", " ").split()]
                if len(row) != 9:
                    raise ValueError(f"Row has {len(row)} values, expected 9")
                puzzle.append(row)

            if len(puzzle) != 9:
                raise ValueError(f"Got {len(puzzle)} rows, expected 9")

            # Solve
            solver = SudokuSolver()
            solved, original = solver.solve(puzzle)

            st.success("✅ Puzzle solved!")

            # Display results in columns
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("Original Puzzle")
                img_original = solver.render_grid(original)
                st.image(img_original, use_container_width=True)

            with res_col2:
                st.subheader("Solution (Blue = Filled)")
                img_solved = solver.render_grid(original, solution=solved)
                st.image(img_solved, use_container_width=True)

            # Download buttons
            st.subheader("📥 Download")
            col1, col2, col3 = st.columns(3)

            buf1 = BytesIO()
            img_original.save(buf1, format="PNG")
            col1.download_button(
                "Original",
                buf1.getvalue(),
                "original_puzzle.png",
                "image/png",
            )

            buf2 = BytesIO()
            img_solved.save(buf2, format="PNG")
            col2.download_button(
                "Solved",
                buf2.getvalue(),
                "solved_puzzle.png",
                "image/png",
            )

            # Solution as array
            solution_text = "\n".join([" ".join(map(str, row)) for row in solved])
            col3.download_button(
                "Solution (TXT)",
                solution_text,
                "solution.txt",
                "text/plain",
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")

    if col2.button("📋 Load Sample", key="load_sample"):
        st.toast("Sample puzzle loaded!")
        st.rerun()

with tab3:
    st.subheader("About")
    st.write(
        """
    **Sudoku Solver** combines OpenCV image processing with backtracking algorithm.

    ### Algorithm
    - **Backtracking**: Tries digits 1-9 in each empty cell, validates against row/column/3×3 box constraints
    - **Image Processing**: Grayscale → Threshold → Contour detection → Perspective warp → Cell split

    ### Features
    - Solve from 9×9 array input
    - Extract grid from Sudoku images (preprocessing visible)
    - Render solutions with original/overlay views
    - Download results as PNG or TXT

    ### Limitations
    - Digit extraction is a placeholder (returns 0 for all cells)
    - Requires relatively clean, well-lit images for preprocessing
    - No handwriting recognition yet

    ### Extend It
    Replace `extract_digit_from_cell()` in `image_processor.py` with:
    - **Pytesseract**: `pip install pytesseract` + Tesseract binary
    - **TensorFlow**: Train on MNIST-like digit dataset
    """
    )

    st.subheader("Sample Puzzle")
    st.code("\n".join([" ".join(map(str, row)) for row in SAMPLE_PUZZLE]))
