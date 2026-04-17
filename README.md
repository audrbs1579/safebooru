# safebooru

This project is now focused on interactive visualization for the Danbooru parquet dataset.

## Project Structure

- `code/1.view.ipynb`
  Interactive notebook for exploring `data/parquet/danbooru.parquet`.
- `data/parquet/danbooru.parquet`
  Main dataset used by the notebook.
- `data/train`, `data/val`, `data/test`
  Kept as-is. They are not modified by the notebook.

## What The Notebook Shows

- Rating distribution
- File extension distribution
- Deleted / banned / active post status
- Interactive scatter plots for score, favorites, tags, size, and resolution
- Top tags by tag category
- Monthly trend charts
- Correlation heatmap for numeric columns
- Source / Pixiv / parent relationship coverage

## Main Libraries

- `polars`
- `pandas`
- `plotly`
- `ipywidgets`
- `numpy`

## Notes

- The notebook reads from `data/parquet/danbooru.parquet`.
- The notebook samples rows for faster exploration instead of loading the full file into memory.
- This repository is organized for visualization work, not crawling.
