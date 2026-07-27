"""
Image Processor - Extract Sudoku grid from image using OpenCV.

Preprocessing pipeline:
1. Grayscale conversion
2. Threshold to binary
3. Find largest contour (Sudoku grid)
4. Perspective warp to flat top-down view
5. Split into 81 cells
6. Extract digits (placeholder for OCR/ML)
"""

import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    """Process Sudoku images to extract grids."""

    @staticmethod
    def preprocess(img_path):
        """
        Preprocess image: grayscale, threshold, find grid contour.
        Returns preprocessed image and contour of Sudoku grid.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold works better for varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return gray, thresh, img

    @staticmethod
    def find_grid_contour(thresh):
        """Find largest contour (assumed to be Sudoku grid)."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in image.")

        largest = max(contours, key=cv2.contourArea)
        return largest

    @staticmethod
    def get_perspective_transform(contour, img_shape):
        """
        Find 4 corners of contour and return warp transform.
        Assumes contour is roughly rectangular.
        """
        eps = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) != 4:
            raise ValueError(
                f"Expected 4 corners, got {len(approx)}. Try adjusting epsilon."
            )

        # Sort corners: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        # Destination: square grid
        dst = np.array(
            [[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32
        )

        matrix = cv2.getPerspectiveTransform(rect, dst)
        return matrix

    @staticmethod
    def warp_grid(img, matrix):
        """Warp image using perspective transform."""
        warped = cv2.warpPerspective(img, matrix, (450, 450))
        return warped

    @staticmethod
    def split_cells(warped_img):
        """
        Split warped 450x450 grid into 9x9=81 cells.
        Each cell is 50x50 pixels.
        Returns list of 81 cell images.
        """
        cells = []
        cell_size = 50

        for i in range(9):
            for j in range(9):
                y = i * cell_size
                x = j * cell_size
                cell = warped_img[y : y + cell_size, x : x + cell_size]
                cells.append(cell)

        return cells

    @staticmethod
    def extract_digit_from_cell(cell_img):
        """
        Extract digit from single cell using OCR (pytesseract).
        Falls back to contour-based detection if OCR unavailable.
        Returns 0 for empty cells.

        Note: pytesseract is optional. Install with: pip install pytesseract
        Requires Tesseract binary: brew install tesseract (macOS) or apt install tesseract (Linux)
        """
        # Convert to grayscale
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Threshold
        _, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)

        # Find contours (digit)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0  # Empty cell

        # Check if any significant contour (not noise)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # If contour too small, likely empty
        if area < 50:
            return 0

        # Try OCR with pytesseract (optional dependency)
        try:
            import pytesseract

            # Upscale for better OCR
            upscaled = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                upscaled, config="--psm 10 -c tessedit_char_whitelist=0123456789"
            )
            digit = int(text.strip()) if text.strip().isdigit() else 0
            return digit if 0 <= digit <= 9 else 0

        except ImportError:
            # pytesseract not installed - use contour fallback
            return ImageProcessor._detect_digit_by_contour(gray, thresh)
        except Exception:
            # Tesseract binary not available or other error
            return 0

    @staticmethod
    def _detect_digit_by_contour(gray, thresh):
        """Fallback digit detection using contour analysis. Returns 0 (empty) if unreliable."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Very conservative: only accept if reasonable digit size range
        # Most printed digits in a 50x50 cell are 200-800 pixels
        if area < 100 or area > 1000:
            return 0  # Too small (noise) or too large

        # Still can't reliably identify which digit without OCR
        # Return 0 to indicate unknown
        return 0

    @classmethod
    def extract_grid_from_image(cls, img_path):
        """
        Full pipeline: load → preprocess → find grid → warp → split → extract digits.
        Returns 9x9 grid (0 for empty cells).
        """
        gray, thresh, img = cls.preprocess(img_path)

        contour = cls.find_grid_contour(thresh)

        matrix = cls.get_perspective_transform(contour, img.shape)

        warped = cls.warp_grid(thresh, matrix)  # Use binary image

        cells = cls.split_cells(warped)

        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                digit = cls.extract_digit_from_cell(cells[i * 9 + j])
                row.append(digit)
            grid.append(row)

        return np.array(grid), warped

    @staticmethod
    def save_preprocessing_stages(img_path, output_dir="output"):
        """Save intermediate images for debugging."""
        processor = ImageProcessor()
        gray, thresh, img = processor.preprocess(img_path)

        cv2.imwrite(f"{output_dir}/stage_01_grayscale.png", gray)
        cv2.imwrite(f"{output_dir}/stage_02_threshold.png", thresh)

        print(f"Saved preprocessing stages to {output_dir}/")


if __name__ == "__main__":
    # Example usage (requires actual Sudoku image)
    print("ImageProcessor module for Sudoku image preprocessing.")
    print("Use: grid, warped = ImageProcessor.extract_grid_from_image('sudoku.jpg')")
