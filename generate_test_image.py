"""Generate a test Sudoku image for testing the extraction pipeline."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def generate_sudoku_image(puzzle, filename="test_sudoku.jpg", cell_size=80):
    """Generate a clean, high-contrast Sudoku grid image."""
    size = 9 * cell_size + 20
    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)

    try:
        # Use larger, clearer font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except:
        font = ImageFont.load_default()

    border_color = "black"
    thick_border = 4
    thin_border = 2

    # Draw grid lines with better spacing
    for i in range(10):
        line_width = thick_border if i % 3 == 0 else thin_border
        x = 10 + i * cell_size
        draw.line([(x, 10), (x, size - 10)], fill=border_color, width=line_width)
        y = 10 + i * cell_size
        draw.line([(10, y), (size - 10, y)], fill=border_color, width=line_width)

    # Draw numbers with high contrast
    for i in range(9):
        for j in range(9):
            x = 10 + j * cell_size + cell_size // 2
            y = 10 + i * cell_size + cell_size // 2
            num = puzzle[i][j]

            if num != 0:
                draw.text((x, y), str(num), fill="black", font=font, anchor="mm")

    img.save(filename, quality=95)
    print(f"✅ Generated test image: {filename} (size: {size}x{size})")
    return filename


if __name__ == "__main__":
    generate_sudoku_image(SAMPLE_PUZZLE, "test_sudoku.jpg")
