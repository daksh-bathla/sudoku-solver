# Streamlit Web App

Interactive web interface for Sudoku Solver.

## Run

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

## Features

### 📋 Array Input Tab
- Paste 9×9 puzzle (0 = empty)
- Space or comma separated values
- Click "Solve" to solve
- Download as PNG/TXT

### 🖼️ Image Upload Tab
- Upload Sudoku grid photo (PNG/JPG)
- See preprocessing stages (grayscale, threshold)
- Extract grid (shows warped, contour-detected result)
- Edit extracted digits before solving
- Download results

### ℹ️ Info Tab
- Algorithm explanation
- Extension guides (pytesseract, TensorFlow)
- Limitations & roadmap

## Example Usage

1. **Array Input**:
   - Paste sample or custom puzzle
   - Click "Solve"
   - Download original/overlay/solution

2. **Image Upload**:
   - Take/upload photo of Sudoku grid
   - Review preprocessing
   - Extract → Edit (if needed) → Solve
   - Download images

## Deployment

### Streamlit Cloud
```bash
git push
# Go to https://share.streamlit.io
# Connect GitHub repo
# Select streamlit_app.py as main file
```

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

```bash
docker build -t sudoku-solver .
docker run -p 8501:8501 sudoku-solver
```

### Heroku
Requires `Procfile`:
```
web: streamlit run streamlit_app.py --logger.level=error --server.port=$PORT --server.address=0.0.0.0
```
