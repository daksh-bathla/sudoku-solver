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
    "Solve Sudoku puzzles from images or hardcoded arrays using backtracking + OpenCV preprocessing."
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
tab1, tab2, tab3 = st.tabs(["📋 Array Input", "🖼️ Image Upload", "ℹ️ Info"])

with tab1:
    st.subheader("Enter Sudoku Puzzle (9×9 Array)")
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
                "Overlay",
                buf2.getvalue(),
                "solved_overlay.png",
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

with tab2:
    st.subheader("Upload Sudoku Image")
    st.write("PNG/JPG image of a Sudoku grid. Preprocessing shows intermediate stages.")

    uploaded_file = st.file_uploader("Choose image...", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Display uploaded image
            st.subheader("Uploaded Image")
            uploaded_img = Image.open(uploaded_file)
            st.image(uploaded_img, use_container_width=True)

            # Preprocessing
            st.subheader("🔍 Preprocessing Stages")

            gray, thresh, original_img = ImageProcessor.preprocess(tmp_path)

            proc_col1, proc_col2 = st.columns(2)

            with proc_col1:
                st.write("**Grayscale**")
                st.image(gray, use_container_width=True, channels="GRAY")

            with proc_col2:
                st.write("**Threshold (Binary)**")
                st.image(thresh, use_container_width=True, channels="GRAY")

            # Extract grid
            if st.button("🔍 Extract Grid", key="extract_grid"):
                try:
                    grid, warped = ImageProcessor.extract_grid_from_image(tmp_path)

                    st.subheader("Extracted Grid (Warped)")
                    st.image(warped, use_container_width=True, channels="GRAY")

                    st.info(
                        "ℹ️ Digit extraction is a placeholder (returns 0 for all cells). "
                        "Integrate pytesseract or TensorFlow to enable OCR."
                    )

                    st.subheader("Extracted Grid Array")
                    st.write(grid)

                    # Option to manually edit grid before solving
                    st.subheader("📝 Edit Grid Before Solving")
                    st.write("Replace 0s with correct digits if needed:")

                    edited_text = st.text_area(
                        "Edit grid:",
                        value="\n".join(
                            [" ".join(map(str, row)) for row in grid]
                        ),
                        height=150,
                    )

                    if st.button("🔍 Solve Extracted", key="solve_extracted"):
                        try:
                            lines = edited_text.strip().split("\n")
                            puzzle = []
                            for line in lines:
                                row = [int(x) for x in line.replace(",", " ").split()]
                                puzzle.append(row)

                            solver = SudokuSolver()
                            solved, original = solver.solve(puzzle)

                            st.success("✅ Solved!")

                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.write("**Original**")
                                img_orig = solver.render_grid(original)
                                st.image(img_orig, use_container_width=True)

                            with res_col2:
                                st.write("**Solution Overlay**")
                                img_sol = solver.render_grid(original, solution=solved)
                                st.image(img_sol, use_container_width=True)

                            # Download
                            buf1 = BytesIO()
                            img_orig.save(buf1, format="PNG")
                            st.download_button(
                                "Download Original",
                                buf1.getvalue(),
                                "original.png",
                                "image/png",
                            )

                            buf2 = BytesIO()
                            img_sol.save(buf2, format="PNG")
                            st.download_button(
                                "Download Solution",
                                buf2.getvalue(),
                                "solution.png",
                                "image/png",
                            )

                        except Exception as e:
                            st.error(f"Error: {e}")

                except Exception as e:
                    st.error(f"Grid extraction failed: {e}")

        finally:
            os.unlink(tmp_path)

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
