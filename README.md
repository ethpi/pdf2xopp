## pdf2xopp

A Python tool to convert PDF/SVG files to Xournal++ format (`.xopp`).

### Features

- Converts vector PDFs (GoodNotes, OneNote, Xournal++) to editable Xournal++ strokes
- Two handwriting modes:
  - **filled** (default): ink outlines rendered as filled area strokes
  - **line** (`--line`): ink outlines reduced to a centerline stroke via ellipse centroid heuristic
- Converts raster-only PDFs (e.g. Samsung Notes exported via [sdocx2pdf](https://github.com/squ1dd13/sdocx2pdf/)):
  embedded bitmap preserved as background layer,
  re-vectorized via skeletonization into an editable stroke layer
- Extracts and embeds text spans with font mapping
- Extracts and embeds bitmaps
- Repairs LaTeX diacritic sequences (ä, ö, ü, …) incorrectly extracted by PyMuPDF
- Detects and filters background grid/dots automatically in seperate layer
- Optionally embeds invisible OCR text for full-text search in Xournal++ in white in seperate layer
- Appends pages to existing `.xopp` files


### Requirements

#### Python Dependencies

- Python 3.9+
- [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`) — PDF parsing and text extraction
- [OpenCV](https://opencv.org/) (`cv2`) — image processing for raster PDFs
- [NumPy](https://numpy.org/) (`numpy`) — numerical operations
- [scikit-image](https://scikit-image.org/) (`skimage`) — skeletonization of raster strokes

#### External Tools

- [Inkscape](https://inkscape.org/) — required for SVG input conversion


### Usage

```
python3 ./pdf2xopp.py [-h] [--append FILE.xopp] [--line] [--include-background] [--include-ocr] [--debug N] input.pdf

or (after chmod +x ./pdf2xopp.py)

./pdf2xopp.py [-h] [--append FILE.xopp] [--line] [--include-background] [--include-ocr] [--debug N] input.pdf
```

### Options

| Option | Description |
|---|---|
| `--append FILE.xopp` | Append pages to an existing `.xopp` file (created if missing) |
| `--line` | Use centerline mode for handwriting strokes |
| `--include-background` | Keep background grid lines as a separate layer |
| `--include-ocr` | Embed invisible OCR text for search indexing in white in a separte layer |
| `--debug N` | Enable debug output at verbosity level N |

### Known Limitations
- Because the program code is AI-generated, it is quite possible that everything is somewhat clunky and not very elegant. However, the code works more or less fine for my applications. Feel free to send me any improvement suggestions.

- **Font mapping**: Font name extraction depends on PDF metadata. PDFs converted
  via CUPS (`cupspdf`) lose font metadata; text may be rendered with incorrect
  fonts. Use the original source PDF (e.g. directly from LaTeX) for best results.
- **Diacritic repair**: The built-in diacritic repair covers the most common
  LaTeX extraction artifacts, but edge cases may remain depending on the PDF
  producer and font encoding used.
- **Vector graphics**: Xournal++ has limited support for complex vector graphics.
  Gradients, clipping paths, transparency, and advanced fill patterns cannot be
  represented faithfully as Xournal++ strokes. Such elements may appear
  incomplete, distorted, or missing in the converted file.


### License

This program is free software licensed under the GNU General Public License v3.0 (or later).

See [LICENSE](LICENSE) for the full license text.

Third-party components and their licenses are listed in [NOTICE](NOTICE).

### Dependencies & Licenses

| Package | License |
|---|---|
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | GNU AGPL 3.0 |
| [OpenCV](https://github.com/opencv/opencv) | Apache 2.0 |
| [NumPy](https://numpy.org/) | BSD 3-Clause |
| [scikit-image](https://scikit-image.org/) | BSD 3-Clause |
| [Inkscape](https://inkscape.org/) | GNU GPL 2.0+ |

# pdf2xopp
