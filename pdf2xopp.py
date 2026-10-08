#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "PyMuPDF",
#   "opencv-python",
#   "numpy",
#   "scikit-image",
# ]
# ///
"""
pdf2xopp.py  –  PDF → Xournal++ (.xopp)

Converts PDF files to Xournal++ format. Handles the following input types:

  1. Vector PDFs (GoodNotes, OneNote, Xournal++): ink stored as filled outline
     paths or stroked line segments. Two handwriting modes are available:
       - filled (default): ink outlines rendered as filled area strokes
       - line (--line):    ink outlines reduced to a centerline stroke via
                           ellipse centroid heuristic
  2. Raster PDFs: embedded bitmap preserved as background layer, then
     re-vectorized via skeletonization into an editable stroke layer.

Background grid lines are detected and filtered out by default.
Use --include-background to keep them.
Use --include-ocr to embed invisible OCR text for search.
Use --append filename.xopp to append pages to an existing .xopp file.
"""

import sys, os, gzip, argparse, base64
import xml.etree.ElementTree as ET
import fitz
import cv2
import numpy as np
from skimage.morphology import skeletonize
import re
import unicodedata
import shutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGLEVEL = 0  # overridden by --debug=N

def log(level, *args, **kwargs):
    if level <= LOGLEVEL:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHITE                 = (1.0, 1.0, 1.0)  # RGB white, used as fill sentinel
GRAY_MIN              = 0.8              # minimum luminance to qualify as gray
BEZIER_STEPS_MAX      = 32               # maximum segments per Bézier curve
BEZIER_STEP_SIZE      = 4.0              # target segment length in pt (lower = more points)
EPS                   = 0.5              # max gap (pt) between points before starting a new subpath
RESOLUTION            = 300              # DPI for raster page rendering
MIN_STROKE            = 3                # minimum pixel length to keep a skeleton stroke
STROKE_WIDTH          = 1.4              # default stroke width (pt) for skeleton strokes
SAT_MIN               = 15               # minimum HSV saturation to count as a colored pixel
DARK_MAX              = 180              # maximum HSV value to count as a dark pixel
GRID_ITEM_THRESHOLD   = 20               # minimum number of lines in a path to suspect a grid
GRID_COLOR_NEUTRALITY = 0.05             # max RGB channel spread to qualify as neutral gray
LUMINANCE_MIN         = 0.65             # minimum luminance to qualify as light background color
THIN                  = 0.6              # threshold (pt): narrower than this → line, not filled area
LINE_WIDTH            = 0.5              # stroke width (pt) used when rendering thin rectangles as lines


# ---------------------------------------------------------------------------
# Vector PDF helpers
# ---------------------------------------------------------------------------

def is_neutral_gray(col):
    """Return True if the color is a neutral gray (not black, not white)."""
    if col is None:
        return False
    r, g, b = col
    diff = max(abs(r-g), abs(g-b), abs(r-b))
    return diff < GRID_COLOR_NEUTRALITY and GRAY_MIN > r > 0.1


def is_light_background_color(col):
    """Return True if the color is light enough to be a background color."""
    if col is None:
        return False
    r, g, b = col
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance > LUMINANCE_MIN


def is_background_fill(p, page_width, page_height):
    """
    Detect fill paths that are part of the background (grid lines, ruled lines,
    or dot grids). Matches:
      1. Full-page background rectangle → 'skip' (recognized but not rendered)
      2. White fills
      3. Neutral grays / light colors spanning the page as thin rectangles
      4. GoodNotes dot grid: small filled circles (4 Bézier arcs, light color, bbox < 20pt)
    """
    col   = p.get('fill')
    items = p.get('items', [])

    # Case 1: Full-page background rectangle → recognized but not rendered
    if col is not None and is_light_background_color(col):
        for item in items:
            if item[0] == 're':
                r = item[1]
                w = r.x1 - r.x0
                h = r.y1 - r.y0
                if w > page_width * 0.9 and h > page_height * 0.9:
                    log(1, f"  [Grid] Full-page bg rect skipped: {r}, color={col}")
                    return 'skip'

    # Case 2: White fills
    if col == WHITE:
        return True

    if col is not None:
        # Case 4: GoodNotes dot grid – small filled circle (4 Bézier arcs)
        if is_light_background_color(col):
            ops = [it[0] for it in items]
            if ops == ['c', 'c', 'c', 'c']:
                bbox = p.get('rect')
                if bbox is not None:
                    w = bbox.x1 - bbox.x0
                    h = bbox.y1 - bbox.y0
                    if w < 20 and h < 20:
                        log(1, f"  [Grid] GoodNotes dot: bbox={bbox}, color={col}")
                        return True

        # Case 3: neutral gray or light color spanning page as thin rectangle
        if is_neutral_gray(col) or is_light_background_color(col):
            for item in items:
                if item[0] == 're':
                    rect         = item[1]
                    width        = rect.x1 - rect.x0
                    height       = rect.y1 - rect.y0
                    spans_width  = width  > page_width  * 0.9
                    spans_height = height > page_height * 0.9
                    is_thin      = height < 5.0 or width < 5.0
                    if (spans_width or spans_height) and is_thin:
                        log(1, f"  [Grid] Fill line: {rect}, color={col}")
                        return True

    return False




def is_background_geometry(p, page_width, page_height):
    """
    Detect stroked paths that are part of the background grid.
    Handles three cases:
      1. GoodNotes: single faint line with low opacity
      2. Xournal++: single gray line spanning the page
      3. Xournal++: many gray lines bundled in one path
    """
    items   = p.get('items', [])
    color   = p.get('color')
    opacity = p.get('stroke_opacity') or 1.0

    # Case 1: GoodNotes – single faint line (any color, low opacity)
    if len(items) == 1 and items[0][0] == 'l' and opacity < 0.3:
        p0 = items[0][1]; p1 = items[0][2]
        dx = abs(p1.x - p0.x); dy = abs(p1.y - p0.y)
        spans = (dx > page_width * 0.9 and dy < 2.0) or \
                (dy > page_height * 0.9 and dx < 2.0)
        if spans:
            log(1, f"  [Grid] GoodNotes faint line "
                   f"(opacity={opacity:.2f}, color={color})")
            return True

    # Case 2: Xournal++ – single gray line spanning the page
    if len(items) == 1 and items[0][0] == 'l' and \
       (is_neutral_gray(color) or is_light_background_color(color)):
        p0 = items[0][1]; p1 = items[0][2]
        dx = abs(p1.x - p0.x); dy = abs(p1.y - p0.y)
        spans = (dx > page_width * 0.9 and dy < 2.0) or \
                (dy > page_height * 0.9 and dx < 2.0)
        if spans:
            log(1, f"  [Grid] Xournal++ gray line: "
                   f"dx={dx:.1f}, dy={dy:.1f}, color={color}")
            return True

    # Case 3: Xournal++ – many gray lines bundled in one path
    if is_neutral_gray(color) or is_light_background_color(color):
        line_items = [it for it in items if it[0] == 'l']
        if len(line_items) >= GRID_ITEM_THRESHOLD:
            spanning = 0
            for it in line_items:
                p0 = it[1]; p1 = it[2]
                dx = abs(p1.x - p0.x); dy = abs(p1.y - p0.y)
                if (dx > page_width * 0.9 and dy < 2.0) or \
                   (dy > page_height * 0.9 and dx < 2.0):
                    spanning += 1
            ratio = spanning / len(line_items)
            if ratio > 0.5:
                log(1, f"  [Grid] Xournal++ grid path: "
                       f"{len(line_items)} lines, "
                       f"{spanning} spanning ({ratio:.0%}), color={color}")
                return True

    return False


def color_to_hex(rgb, opacity=1.0):
    """
    Convert an RGB tuple (0.0–1.0 each) and opacity to an 8-digit hex color
    string (#rrggbbff). Opacity is blended against white.
    """
    if rgb is None:
        return "#000000ff"
    r, g, b = [max(0, min(255, int(c * 255))) for c in rgb]
    if opacity < 1.0:
        r = int(r * opacity + 255 * (1 - opacity))
        g = int(g * opacity + 255 * (1 - opacity))
        b = int(b * opacity + 255 * (1 - opacity))
    return f"#{r:02x}{g:02x}{b:02x}ff"


def pt_dist(a, b):
    """Euclidean distance between two 2D points."""
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def bezier_pts(p0, p1, p2, p3, steps=None):
    """
    Sample a cubic Bézier curve defined by four control points.
    Returns a list of (x, y) tuples. Steps are chosen adaptively
    based on the chord length, clamped to [4, BEZIER_STEPS_MAX].
    """
    if steps is None:
        chord = pt_dist(p0, p3)
        steps = max(4, min(BEZIER_STEPS_MAX, int(chord / BEZIER_STEP_SIZE)))
    pts = []
    for i in range(steps + 1):
        t = i / steps; mt = 1 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts



def split_into_subpaths(items, page_height):
    """
    Split a PyMuPDF items list into separate subpaths.
    A new subpath starts whenever there is a gap between the end of the
    previous segment and the start of the next one (distance > EPS).
    Rectangle items ('re') are converted to horizontal midpoint lines.
    Returns a list of point lists.
    """
    def fpt(p): return (p.x, p.y)
    subpaths = []; current = []; last_end = None
    for item in items:
        op = item[0]
        if op == 'l':
            p0 = fpt(item[1]); p1 = fpt(item[2])
            if last_end is None or pt_dist(p0, last_end) > EPS:
                if current: subpaths.append(current)
                current = [p0]
            current.append(p1); last_end = p1
        elif op == 'c':
            p0=fpt(item[1]); p1=fpt(item[2]); p2=fpt(item[3]); p3=fpt(item[4])
            if last_end is None or pt_dist(p0, last_end) > EPS:
                if current: subpaths.append(current)
                current = []
                current.extend(bezier_pts(p0, p1, p2, p3))
            else:
                current.extend(bezier_pts(p0, p1, p2, p3)[1:])
            last_end = p3
        elif op == 're':
            r = item[1]; cy = (r.y0 + r.y1) / 2
            if current: subpaths.append(current)
            subpaths.append([(r.x0, cy), (r.x1, cy)])
            current = []; last_end = None
    if current: subpaths.append(current)
    return subpaths


def subpath_centroid(pts):
    """Return the arithmetic centroid of a list of 2D points."""
    if not pts: return None
    return (sum(p[0] for p in pts)/len(pts),
            sum(p[1] for p in pts)/len(pts))


def subpath_radius(pts, center):
    """
    Return the minimum distance from the center to any point in the subpath.
    Used as a rough radius estimate for the ellipse centroid heuristic.
    """
    if not pts or center is None: return 1.0
    return max(0.3, sorted([pt_dist(p, center) for p in pts])[0])


def signed_area(pts):
    """
    Compute the signed area of a polygon via the Shoelace formula.
    Positive = CCW winding, negative = CW winding (in PDF coordinates
    where Y increases downward, the sign is inverted relative to screen).
    Used to detect inner (hole) subpaths in donut-shaped filled paths:
    a subpath whose winding direction differs from the outer subpath is a hole.
    Note: Xournal++ has no native even-odd or clipping support, so holes
    are approximated by filling them white. This works correctly only when
    no later path overlaps the hole area.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += (x0 * y1) - (x1 * y0)
    return a / 2.0


def points_to_xopp(points):
    """Serialize a list of (x, y) tuples to the Xournal++ stroke point format."""
    return " ".join(f"{v:.4f}" for pt in points for v in pt)


# ---------------------------------------------------------------------------
# Xournal++ element builders
# ---------------------------------------------------------------------------

def make_stroke(layer_el, color, width, pts, widths=None):
    """
    Append a <stroke tool="pen"> element to the layer.
    If `widths` is provided, it is used as per-segment pressure data.
    The widths list is trimmed or padded to match len(pts)-1 as required
    by the Xournal++ format.
    """
    s = ET.SubElement(layer_el, "stroke")
    s.set("tool",     "pen")
    s.set("color",    color)
    s.set("capStyle", "round")

    eff_widths = None
    if widths:
        n_points = len(pts)
        expected = max(0, n_points - 1)
        k = len(widths)

        if k == expected:
            eff_widths = widths
        elif k > expected:
            # Too many pressure points – trim to expected length
            eff_widths = widths[:expected]
        elif k < expected and k > 0:
            # Too few – pad with the last value
            eff_widths = widths + [widths[-1]] * (expected - k)
        # If k == 0, leave eff_widths = None (no pressure data)

    if eff_widths and len(eff_widths) > 0:
        s.set("width", f"{width:.4f} " + " ".join(f"{w:.4f}" for w in eff_widths))
    else:
        # Constant width only
        s.set("width", f"{width:.4f}")

    s.text = points_to_xopp(pts)


def make_filled_stroke(layer_el, color, pts):
    """
    Append a filled <stroke> element (fill=255, width=0) to the layer.
    Used for filled shapes and handwriting outlines.
    """
    s = ET.SubElement(layer_el, "stroke")
    s.set("tool",     "pen")
    s.set("color",    color)
    s.set("capStyle", "round")
    s.set("fill",     "255")
    s.set("width",    "0.0000")
    s.text = points_to_xopp(pts)


def make_highlighter(layer_el, color, pts):
    """Append a <stroke tool="highlighter"> element with 50% transparency."""
    s = ET.SubElement(layer_el, "stroke")
    s.set("tool",  "highlighter")
    s.set("color", color)
    s.set("fill",  "120")
    s.set("width", "10.0000")
    s.text = points_to_xopp(pts)


def rect_to_strokes(layer_el, color, width, rect):
    """Render a rectangle as four individual edge strokes."""
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    edges = [
        [(x0, y0), (x1, y0)],
        [(x1, y0), (x1, y1)],
        [(x1, y1), (x0, y1)],
        [(x0, y1), (x0, y0)],
    ]
    for pts in edges:
        make_stroke(layer_el, color, width, pts)


def classify_rect_item(rect):
    """
    Classify a rectangle item by its dimensions:
      'corner' – both sides thinner than THIN (dot/corner artifact)
      'hline'  – height < THIN (horizontal line)
      'vline'  – width  < THIN (vertical line)
      'area'   – filled rectangle
    """
    w, h = rect.width, rect.height
    if w <= THIN and h <= THIN:
        return 'corner'
    elif h <= THIN:
        return 'hline'
    elif w <= THIN:
        return 'vline'
    else:
        return 'area'


def is_gray_fill(fill):
    """Return True if fill is a neutral gray (not black, not white)."""
    if fill is None or fill == WHITE or fill == (0.0, 0.0, 0.0):
        return False
    r, g, b = fill
    lum  = 0.299*r + 0.587*g + 0.114*b
    diff = max(abs(r-g), abs(g-b), abs(r-b))
    return lum >= GRAY_MIN and diff < 0.05


def fill_rect_solid(layer_el, hex_color, rect):
    """
    Render a rectangle as a closed filled pen stroke (fill=255).
    The rectangle corners are traced as a closed polygon.
    """
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    s = ET.SubElement(layer_el, "stroke")
    s.set("tool",     "pen")
    s.set("color",    hex_color)
    s.set("capStyle", "round")
    s.set("fill",     "255")
    s.set("width",    "0.0000")
    s.text = points_to_xopp(pts)


# ---------------------------------------------------------------------------
# Raster page helpers
# ---------------------------------------------------------------------------

def make_image_layer(page_el, pw, ph, pix):
    """
    Embed a rendered page pixmap as a base64-encoded PNG <image> element
    covering the full page. Used as background for raster PDFs.
    """
    png_bytes = pix.tobytes("png")
    b64       = base64.b64encode(png_bytes).decode('ascii')
    layer_el  = ET.SubElement(page_el, "layer")
    img_el    = ET.SubElement(layer_el, "image")
    img_el.set("left",   "0.0000")
    img_el.set("top",    "0.0000")
    img_el.set("right",  f"{pw:.4f}")
    img_el.set("bottom", f"{ph:.4f}")
    img_el.text = b64


def rgb_to_hex(r, g, b):
    """Convert integer RGB values (0–255) to an 8-digit hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}ff"


def skeleton_to_strokes(skel):
    """
    Convert a binary skeleton image (numpy bool array) to a list of
    pixel-coordinate strokes. Uses a depth-first traversal starting from
    endpoints (pixels with exactly one neighbor) to produce ordered paths.
    Strokes shorter than MIN_STROKE pixels are discarded.
    """
    ys, xs    = np.where(skel)
    pixel_set = set(zip(xs.tolist(), ys.tolist()))
    visited   = set(); strokes = []

    def neighbors(x, y):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x+dx, y+dy
                if (nx, ny) in pixel_set and (nx, ny) not in visited:
                    yield nx, ny

    def degree(x, y):
        return sum(1 for _ in neighbors(x, y))

    endpoints        = [(x, y) for x, y in pixel_set if degree(x, y) == 1]
    start_candidates = endpoints if endpoints else list(pixel_set)

    for start in start_candidates:
        if start in visited: continue
        path = []; stack = [start]
        while stack:
            pt = stack.pop()
            if pt in visited: continue
            visited.add(pt); path.append(pt)
            nbrs = list(neighbors(*pt))
            nbrs.sort(key=lambda p: degree(*p), reverse=True)
            stack.extend(nbrs)
        if len(path) >= MIN_STROKE:
            strokes.append(path)
    return strokes


def smooth_stroke(points, window=5):
    """
    Apply a simple moving-average smoothing to a list of (x, y) points.
    Each point is replaced by the average of its `window` neighbors.
    """
    if len(points) < window: return points
    result = []; half = window // 2
    for i in range(len(points)):
        lo = max(0, i-half); hi = min(len(points), i+half+1)
        xs = [p[0] for p in points[lo:hi]]
        ys = [p[1] for p in points[lo:hi]]
        result.append((sum(xs)/len(xs), sum(ys)/len(ys)))
    return result


def dominant_color(img_bgr, points):
    """
    Sample up to 10 evenly spaced pixels along a stroke and return the
    average color as an 8-digit hex string. Falls back to black if no
    valid pixels are found.
    """
    rs, gs, bs = [], [], []
    for x, y in points[::max(1, len(points)//10)]:
        px, py = int(x), int(y)
        if 0 <= py < img_bgr.shape[0] and 0 <= px < img_bgr.shape[1]:
            b, g, r = img_bgr[py, px]
            rs.append(int(r)); gs.append(int(g)); bs.append(int(b))
    if not rs: return "#000000ff"
    return rgb_to_hex(int(sum(rs)/len(rs)),
                      int(sum(gs)/len(gs)),
                      int(sum(bs)/len(bs)))


def split_stroke_by_color(stroke, img_bgr, max_color_jump=60):
    """
    Split a stroke into segments wherever the pixel color changes abruptly.
    A new segment starts when the L2 distance between adjacent pixel colors
    exceeds `max_color_jump`. Used to separate multi-color skeleton strokes.
    """
    if len(stroke) < 2:
        return [stroke]
    result = []; current = [stroke[0]]
    for pt in stroke[1:]:
        x1, y1 = current[-1]
        x2, y2 = pt
        c1 = img_bgr[y1, x1].astype(float)
        c2 = img_bgr[y2, x2].astype(float)
        diff = np.linalg.norm(c1 - c2)
        if diff > max_color_jump:
            result.append(current)
            current = [pt]
        else:
            current.append(pt)
    result.append(current)
    return result


def make_skeleton_layer(page_el, page, pw, ph, pix=None):
    """
    Rasterize the page, extract dark and saturated pixel masks, skeletonize
    them, and convert the resulting skeleton to Xournal++ strokes.

    Pipeline:
      1. Render page to bitmap at RESOLUTION DPI (or reuse `pix`)
      2. Build two masks: dark pixels (v < DARK_MAX) and saturated pixels (s > SAT_MIN)
      3. Morphological opening to remove noise
      4. Skeletonize each mask → 1-pixel-wide centerlines
      5. Trace skeleton pixels into ordered strokes (skeleton_to_strokes)
      6. Split strokes at color jumps (split_stroke_by_color)
      7. Smooth each stroke (smooth_stroke)
      8. Sample dominant color per stroke (dominant_color)
      9. Scale pixel coordinates back to PDF points and emit <stroke> elements
    """
    scale = 72.0 / RESOLUTION
    if pix is None:
        mat = fitz.Matrix(RESOLUTION / 72, RESOLUTION / 72)
        pix = page.get_pixmap(matrix=mat)

    img_bgr = cv2.imdecode(
        np.frombuffer(pix.tobytes("png"), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img_bgr is None:
        print("    [Skeleton] Could not render page, skipping vector layer.")
        return

    print(f"    Rendered: {pix.width} x {pix.height} px at {RESOLUTION} DPI")

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    kernel  = np.ones((3, 3), np.uint8)

    mask_dark = (v < DARK_MAX).astype(np.uint8) * 255
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel)
    skel_dark = skeletonize((mask_dark > 0).astype(np.uint8))

    mask_sat  = (s > SAT_MIN).astype(np.uint8) * 255
    mask_sat  = cv2.morphologyEx(mask_sat, cv2.MORPH_OPEN, kernel)
    skel_sat  = skeletonize((mask_sat > 0).astype(np.uint8))

    log(1, f"    [Skeleton] Skeleton pixels: "
           f"dark={np.sum(skel_dark)}, sat={np.sum(skel_sat)}")

    raw_strokes  = skeleton_to_strokes(skel_dark)
    raw_strokes += skeleton_to_strokes(skel_sat)
    split_strokes = []
    for raw in raw_strokes:
        split_strokes.extend(split_stroke_by_color(raw, img_bgr))
    raw_strokes = split_strokes

    print(f"    Skeleton strokes: {len(raw_strokes)}")
    for i, r in enumerate(raw_strokes[:10]):
        col = dominant_color(img_bgr, r)
        print(f"      Stroke {i}: {len(r)} px, color={col}")

    layer_el     = ET.SubElement(page_el, "layer")
    stroke_count = 0
    for raw in raw_strokes:
        smoothed = smooth_stroke(raw, window=5)
        if len(smoothed) < MIN_STROKE: continue
        color  = dominant_color(img_bgr, smoothed)
        scaled = [(x * scale, y * scale) for x, y in smoothed]
        make_stroke(layer_el, color, STROKE_WIDTH, scaled)
        stroke_count += 1

    print(f"    {stroke_count} skeleton stroke(s) generated.")


# ---------------------------------------------------------------------------
# Shared helpers for vector page elements
# ---------------------------------------------------------------------------

def extract_image_with_alpha(doc, xref, smask):
    """
    Extract an image from the PDF and composite it with its soft mask (smask)
    to produce a PNG with correct alpha channel. Returns raw PNG bytes.
    If the mask dimensions differ from the image, it is resized to match.
    """
    info = doc.extract_image(xref)
    data = np.frombuffer(info['image'], np.uint8)
    im   = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        return info['image']

    minfo = doc.extract_image(smask)
    mdata = np.frombuffer(minfo['image'], np.uint8)
    mask  = cv2.imdecode(mdata, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        if mask.shape[:2] != im.shape[:2]:
            mask = cv2.resize(mask, (im.shape[1], im.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        if im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        im[:, :, 3] = mask

    _, buf = cv2.imencode('.png', im)
    return buf.tobytes()


def add_images(layer_el, page, pw, ph, doc):
    """
    Embed all images found on the page as <image> elements in the layer.
    Tries to determine the bounding box via get_image_rects(), falling back
    to get_image_bbox() and finally the full page rect.
    Soft-masked images are composited with their alpha channel.
    """
    for img_info in page.get_images(full=True):
        xref     = img_info[0]
        smask    = img_info[1]
        img_name = img_info[7]
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
            else:
                try:
                    b = page.get_image_bbox(img_name)
                    x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
                except Exception:
                    x0, y0, x1, y1 = 0, 0, pw, ph
                    log(1, f"  [Image] xref={xref}: no rect, "
                           f"falling back to full page")

            if smask:
                img_data = extract_image_with_alpha(doc, xref, smask)
            else:
                img_data = doc.extract_image(xref)['image']

            b64    = base64.b64encode(img_data).decode('ascii')
            img_el = ET.SubElement(layer_el, "image")
            img_el.set("left",   f"{x0:.4f}")
            img_el.set("top",    f"{y0:.4f}")
            img_el.set("right",  f"{x1:.4f}")
            img_el.set("bottom", f"{y1:.4f}")
            img_el.text = b64
            log(1, f"  [Image] xref={xref}, smask={smask}, "
                   f"bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")
        except Exception as e:
            log(1, f"  [Image] xref={xref} failed: {e}")


# ---------------------------------------------------------------------------
# Font name mapping: PDF font names → Xournal++-compatible font names
# ---------------------------------------------------------------------------

FONT_MAP = {
    # --- Application-specific (may vary) ---
    "f-0-0": "Sans Bold",
    "f-1-0": "Serif",
    "f-2-0": "Serif Italic",
    "f-7-0": "Sans",
    "F1":    "Sans",
    "Arimo": "Liberation Sans",

    # --- Sans / UI ---
    "Arial":                 "Liberation Sans",
    "ArialMT":               "Liberation Sans",
    "Arial-Bold":            "Liberation Sans Bold",
    "Arial-BoldMT":          "Liberation Sans Bold",
    "Helvetica":             "Liberation Sans",
    "HelveticaNeue":         "Liberation Sans",
    "Calibri":               "Noto Sans",
    "Verdana":               "Liberation Sans",
    "Tahoma":                "Liberation Sans",
    "TrebuchetMS":           "Liberation Sans",
    "SegoeUI":               "Noto Sans",

    # --- Serif ---
    "Times":                 "Liberation Serif",
    "TimesRoman":            "Liberation Serif",
    "Times-Roman":           "Liberation Serif",
    "TimesNewRomanPS":       "Liberation Serif",
    "TimesNewRomanPSMT":     "Liberation Serif",
    "Georgia":               "Noto Serif",
    "Cambria":               "Noto Serif",
    "Palatino":              "TeX Gyre Pagella",
    "URWPalladioL":          "TeX Gyre Pagella",
    "Bookman":               "TeX Gyre Bonum",
    "Utopia":                "TeX Gyre Utopia",
    "ZapfChancery":          "TeX Gyre Chorus",

    # --- Monospace ---
    "Courier":               "Liberation Mono",
    "CourierNew":            "Liberation Mono",
    "CourierNewPSMT":        "Liberation Mono",

    # --- Symbol / Dingbats ---
    "Symbol":                "Noto Sans Symbols",
    "ZapfDingbats":          "Noto Sans Symbols",

    # --- CJK ---
    "PingFang":              "Noto Sans CJK SC",
    "Hiragino":              "Noto Sans CJK JP",
    "MSMincho":              "Noto Serif CJK JP",
    "MSPGothic":             "Noto Sans CJK JP",
    "SimSun":                "Noto Sans CJK SC",
    "SimHei":                "Noto Sans CJK SC",
    "PMingLiU":              "Noto Serif CJK TC",
    "MalgunGothic":          "Noto Sans CJK KR",
    "Gulim":                 "Noto Sans CJK KR",

    # --- Computer Modern (LaTeX) ---
    "CMR":                   "Serif",
    "CMBX":                  "Serif Bold",
    "CMTI":                  "Serif Italic",
    "CMSS":                  "Sans",
    "CMTT":                  "Monospace",
    "CMMI":                  "Serif Italic",
    "CMSY":                  "Serif",

    # --- Nimbus Roman (pslatex / URW) ---
    "NimbusRomNo9L":         "Serif",
    "NimbusRomNo9LRegu":     "Serif",
    "NimbusRomNo9L-Regu":    "Serif",
    "NimbusRomNo9L-Medi":    "Serif Bold",
    "NimbusRomNo9L-ReguItal":"Serif Italic",
    "NimbusRomNo9L-MediItal":"Serif Bold Italic",
    "TeXGyreTermes":         "TeX Gyre Termes",

    # --- Nimbus Sans (pslatex / URW) ---
    "NimbusSanL":            "Sans",
    "NimbusSanL-Regu":       "Sans",
    "NimbusSanL-Bold":       "Sans Bold",
    "NimbusSanL-ReguItal":   "Sans Italic",
    "NimbusSanL-BoldItal":   "Sans Bold Italic",
    "TeXGyreHeros":          "TeX Gyre Heros",

    # --- Nimbus Mono (pslatex / URW) ---
    "NimbusMonL":            "Monospace",
    "NimbusMonL-Regu":       "Monospace",
    "NimbusMonL-Bold":       "Monospace Bold",
    "TeXGyreCursor":         "TeX Gyre Cursor",
}


def generate_fontname(span) -> str:
    """
    Derive a Xournal++-compatible font name from a PyMuPDF span's 'font' field.
    Resolution order:
      1. Exact match in FONT_MAP
      2. Match after stripping trailing digits (e.g. 'CMR12' → 'CMR')
      3. Prefix match (e.g. 'TimesNewRomanPS-BoldMT' → 'TimesNewRomanPS')
      4. Name-based heuristic: mono / sans / serif fallback
    Subset prefixes like 'AAAAAP+FontName' are stripped before lookup.
    """
    raw = span.get("font", "") or ""

    # Strip subset prefix: "AAAAAP+TimesNewRoman-Bold" → "TimesNewRoman-Bold"
    if "+" in raw:
        raw = raw.split("+", 1)[1]

    # Strip trailing digits for lookup: "CMR12" → "CMR"
    raw_stripped = raw.rstrip("0123456789")

    # 1. Exact match
    if raw in FONT_MAP:
        return FONT_MAP[raw]

    # 2. Match without trailing digits
    if raw_stripped in FONT_MAP:
        return FONT_MAP[raw_stripped]

    # 3. Prefix match: "TimesNewRomanPS-BoldMT" hits "TimesNewRomanPS"
    for key in FONT_MAP:
        if raw.startswith(key):
            return FONT_MAP[key]

    # 4. Fallback: name-based heuristic
    name = raw.lower()
    if any(x in name for x in ("mono", "courier", "consolas", "inconsolata",
                                "sourcecodepro", "dejavumono", "hack",
                                "firacode", "jetbrains")):
        return "Monospace"
    elif any(x in name for x in ("sans", "arial", "helvetica", "calibri",
                                 "gill", "futura", "myriad", "tahoma",
                                 "verdana", "trebuchet")):
        return "Sans"
    else:
        return "Serif"


# ---------------------------------------------------------------------------
# LaTeX diacritics repair map
# Fixes incorrectly extracted characters from LaTeX-generated PDFs
# ---------------------------------------------------------------------------

LATEX_DIACRITICS_MAP = {
    # Umlaut / Diaeresis  ¨ (U+00A8)
    '¨a': 'ä', '¨o': 'ö', '¨u': 'ü', '¨e': 'ë', '¨i': 'ï', '¨y': 'ÿ',
    '¨A': 'Ä', '¨O': 'Ö', '¨U': 'Ü', '¨E': 'Ë', '¨I': 'Ï',
    # Grave  ` (U+0060)
    '`a': 'à', '`e': 'è', '`i': 'ì', '`o': 'ò', '`u': 'ù',
    '`A': 'À', '`E': 'È', '`I': 'Ì', '`O': 'Ò', '`U': 'Ù',
    # Acute  ´ (U+00B4)
    '´a': 'á', '´e': 'é', '´i': 'í', '´o': 'ó', '´u': 'ú', '´y': 'ý',
    '´c': 'ć', '´n': 'ń', '´s': 'ś', '´z': 'ź',
    '´A': 'Á', '´E': 'É', '´I': 'Í', '´O': 'Ó', '´U': 'Ú', '´Y': 'Ý',
    '´C': 'Ć', '´N': 'Ń', '´S': 'Ś', '´Z': 'Ź',
    # Circumflex  ^ (U+005E)
    '^a': 'â', '^e': 'ê', '^i': 'î', '^o': 'ô', '^u': 'û',
    '^A': 'Â', '^E': 'Ê', '^I': 'Î', '^O': 'Ô', '^U': 'Û',
    # Tilde  ~ (U+007E)
    '~a': 'ã', '~n': 'ñ', '~o': 'õ',
    '~A': 'Ã', '~N': 'Ñ', '~O': 'Õ',
    # Cedilla  ¸ (U+00B8)
    '¸c': 'ç', '¸C': 'Ç',
    # Ring  ° (U+00B0)
    '°a': 'å', '°A': 'Å',
    # Caron / Háček  ˇ (U+02C7)
    'ˇc': 'č', 'ˇs': 'š', 'ˇz': 'ž', 'ˇn': 'ň', 'ˇr': 'ř',
    'ˇC': 'Č', 'ˇS': 'Š', 'ˇZ': 'Ž', 'ˇN': 'Ň', 'ˇR': 'Ř',
    # Macron  ¯ (U+00AF)
    '¯a': 'ā', '¯e': 'ē', '¯i': 'ī', '¯o': 'ō', '¯u': 'ū',
    '¯A': 'Ā', '¯E': 'Ē', '¯I': 'Ī', '¯O': 'Ō', '¯U': 'Ū',

    # Combining characters (U+0300–U+036F) — XeLaTeX / LuaLaTeX with modern fonts
    'a\u0308': 'ä', 'o\u0308': 'ö', 'u\u0308': 'ü',
    'e\u0308': 'ë', 'i\u0308': 'ï', 'y\u0308': 'ÿ',
    'A\u0308': 'Ä', 'O\u0308': 'Ö', 'U\u0308': 'Ü',
    'E\u0308': 'Ë', 'I\u0308': 'Ï',
    'a\u0300': 'à', 'e\u0300': 'è', 'i\u0300': 'ì',
    'o\u0300': 'ò', 'u\u0300': 'ù',
    'A\u0300': 'À', 'E\u0300': 'È', 'I\u0300': 'Ì',
    'O\u0300': 'Ò', 'U\u0300': 'Ù',
    'a\u0301': 'á', 'e\u0301': 'é', 'i\u0301': 'í',
    'o\u0301': 'ó', 'u\u0301': 'ú', 'y\u0301': 'ý',
    'A\u0301': 'Á', 'E\u0301': 'É', 'I\u0301': 'Í',
    'O\u0301': 'Ó', 'U\u0301': 'Ú', 'Y\u0301': 'Ý',
    'a\u0302': 'â', 'e\u0302': 'ê', 'i\u0302': 'î',
    'o\u0302': 'ô', 'u\u0302': 'û',
    'A\u0302': 'Â', 'E\u0302': 'Ê', 'I\u0302': 'Î',
    'O\u0302': 'Ô', 'U\u0302': 'Û',
    'a\u0303': 'ã', 'n\u0303': 'ñ', 'o\u0303': 'õ',
    'A\u0303': 'Ã', 'N\u0303': 'Ñ', 'O\u0303': 'Õ',
    'c\u0327': 'ç', 'C\u0327': 'Ç',
    'a\u030a': 'å', 'A\u030a': 'Å',
    'c\u030c': 'č', 's\u030c': 'š', 'z\u030c': 'ž',
    'n\u030c': 'ň', 'r\u030c': 'ř',
    'C\u030c': 'Č', 'S\u030c': 'Š', 'Z\u030c': 'Ž',
    'N\u030c': 'Ň', 'R\u030c': 'Ř',
    'a\u0304': 'ā', 'e\u0304': 'ē', 'i\u0304': 'ī',
    'o\u0304': 'ō', 'u\u0304': 'ū',
    'A\u0304': 'Ā', 'E\u0304': 'Ē', 'I\u0304': 'Ī',
    'O\u0304': 'Ō', 'U\u0304': 'Ū',
}



def fix_latex_diacritics(text):
    """Replace incorrectly extracted LaTeX diacritic sequences with correct Unicode."""
    import unicodedata
    for bad, good in LATEX_DIACRITICS_MAP.items():
        text = text.replace(bad, good)
    return unicodedata.normalize('NFC', text)


def add_texts(layer_el, page, page_el, include_ocr):
    """
    Extract text spans from the page and emit <text> elements.

    - Normal text is added to `layer_el` with a resolved Xournal++ font name.
    - OCR text (font name starting with '.') is either suppressed or added
      as invisible white text to a separate layer inserted at the bottom of
      the page (for search indexing only).
    - LaTeX diacritic sequences are repaired before output.

    Returns (added, suppressed) counts.
    """
    blocks     = page.get_text("rawdict",
                               flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    added      = 0
    suppressed = 0

    # Prepare a separate OCR layer (created lazily on first OCR span)
    ocr_layer_el = None

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):

                # Extract text content
                text = span.get("text") or "".join(
                    c.get("c", "") for c in span.get("chars", []))
                text = fix_latex_diacritics(text)
                text = text.strip()
                if not text:
                    continue

                bbox = span.get("bbox")
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox

                size      = span.get("size", 12.0)
                color_int = span.get("color", 0)
                color_hex = "#{:02x}{:02x}{:02x}ff".format(
                    (color_int >> 16) & 0xFF,
                    (color_int >>  8) & 0xFF,
                     color_int        & 0xFF)

                # Detect OCR text by font name starting with '.'
                raw_font = span.get("font", "") or ""
                if raw_font.lstrip().startswith("."):
                    if not include_ocr:
                        suppressed += 1
                        continue
                    # Create OCR layer on first OCR span
                    if ocr_layer_el is None:
                        ocr_layer_el = ET.Element("layer")
                        page_el.insert(0, ocr_layer_el)
                    text_el = ET.SubElement(ocr_layer_el, "text")
                    text_el.set("font",  "Sans")
                    text_el.set("size",  f"{size:.2f}")
                    text_el.set("x",     f"{x0:.2f}")
                    text_el.set("y",     f"{y0:.2f}")
                    text_el.set("color", "#ffffffff")  # invisible white
                    text_el.text = text
                    suppressed += 1
                    continue

                # Normal text: resolve font name
                font = generate_fontname(span)

                text_el = ET.SubElement(layer_el, "text")
                text_el.set("font",  font)
                text_el.set("size",  f"{size:.2f}")
                text_el.set("x",     f"{x0:.2f}")
                text_el.set("y",     f"{y0:.2f}")
                text_el.set("color", color_hex)
                text_el.text = text
                log(1, f'    [Text] pdf="{span.get("font", "")}" → font="{font}", '
                       f'size="{size:.2f}", color="{color_hex}", '
                       f'flags={span.get("flags", 0):05b}, '
                       f'string="{text[:40]}"')
                added += 1

    return added, suppressed


def path_to_pts(items, eps=EPS):
    """
    Convert a PyMuPDF items list (ops 'l', 'c', 're') to a list of point lists.
    Each gap larger than `eps` between the end of one segment and the start
    of the next starts a new sublist. Rectangle items ('re') are emitted as
    closed four-corner polygons.
    """
    segments = []
    pts = []
    last = None
    for item in items:
        op = item[0]
        if op == 'l':
            p0 = (item[1].x, item[1].y)
            p1 = (item[2].x, item[2].y)
            if last is None or pt_dist(p0, last) > eps:
                if pts:
                    segments.append(pts)
                pts = [p0, p1]
            else:
                pts.append(p1)
            last = p1
        elif op == 'c':
            p0 = (item[1].x, item[1].y)
            p1 = (item[2].x, item[2].y)
            p2 = (item[3].x, item[3].y)
            p3 = (item[4].x, item[4].y)
            curve = bezier_pts(p0, p1, p2, p3)
            if last is None or pt_dist(p0, last) > eps:
                if pts:
                    segments.append(pts)
                pts = curve
            else:
                pts.extend(curve[1:])
            last = p3
        elif op == 're':
            if pts:
                segments.append(pts)
                pts = []
            # Rectangle as closed four-corner polygon
            r = item[1]
            segments.append([
                (r.x0, r.y0), (r.x1, r.y0),
                (r.x1, r.y1), (r.x0, r.y1),
                (r.x0, r.y0)
            ])
            last = None
        elif op == 'qu':
            if pts:
                segments.append(pts)
                pts = []
            # Quad as closed four-corner polygon (PyMuPDF >= 1.22)
            q = item[1]
            segments.append([
                (q.ul.x, q.ul.y),
                (q.ur.x, q.ur.y),
                (q.lr.x, q.lr.y),
                (q.ll.x, q.ll.y),
                (q.ul.x, q.ul.y)   # close the shape
            ])
            last = None
    if pts:
        segments.append(pts)
    return segments



def add_geometry(layer_el, drawings, pw, ph, pass_num=None):
    """
    Render all geometry paths of a page to Xournal++ stroke elements.

    Paths are processed in four passes to ensure correct z-ordering:

      Pass 1  – Gray filled rectangles (rendered first, below everything)
      Pass 2a – Stroked paths (color set, no fill)
      Pass 2b – Rectangle-only fill paths: classified as hline / vline / area
      Pass 2c – Fill+stroke paths (type='fs'): filled area + outline stroke
      Pass 2d – Fill-only curve paths (type='f'): filled shapes.
                 Donut detection via winding direction (Shoelace formula):
                 inner subpaths with opposite winding are filled white.
                 Warning: Xournal++ has no clipping/even-odd support, so
                 white holes may incorrectly cover later overlapping paths.

    pass_num=1    run Pass 1 only  (call before text/images)
    pass_num=2    run Passes 2a–2d (call after text/images)
    pass_num=None run all passes   (backward compatibility)
    """

    rect_only_paths = [
        p for p in drawings
        if p.get('color') is None
        and p.get('fill') is not None
        and p.get('fill') != WHITE
        and all(it[0] == 're' for it in p.get('items', []))
        and not is_background_fill(p, pw, ph)
    ]

    stroked_paths = [
        p for p in drawings
        if p.get('color') is not None
        and p.get('fill') is None
    ]

    # Fill+stroke paths (type='fs') – outline + filled area
    filled_stroked_paths = [
        p for p in drawings
        if p.get('color') is not None
        and p.get('fill') is not None
        and p.get('fill') != WHITE
        and not is_background_fill(p, pw, ph)
    ]

    # Fill-only curve paths (type='f', not exclusively 're' items)
    filled_curve_paths = [
        p for p in drawings
        if p.get('color') is None
        and p.get('fill') is not None
        and p.get('fill') != WHITE
        and not all(it[0] == 're' for it in p.get('items', []))
        and not is_background_fill(p, pw, ph)
    ]

    stroke_count = 0

    # ------------------------------------------------------------------
    # Pass 1: gray filled rectangles (rendered first, below everything)
    # ------------------------------------------------------------------
    if pass_num in (1, None):
        for path in rect_only_paths:
            if not is_gray_fill(path.get('fill')):
                continue
            hex_color = color_to_hex(path['fill'], path.get('fill_opacity') or 1.0)
            for item in path.get('items', []):
                rect = item[1]
                if classify_rect_item(rect) != 'area':
                    continue
                fill_rect_solid(layer_el, hex_color, rect)
                stroke_count += 1
                log(1, f"  [Geo P1] gray area "
                       f"({rect.x0:.1f},{rect.y0:.1f})"
                       f"({rect.x1:.1f},{rect.y1:.1f})")

    # ------------------------------------------------------------------
    # Pass 2a: stroked paths (color set, no fill)
    # ------------------------------------------------------------------
    if pass_num in (2, None):
        for path in stroked_paths:
            items     = path.get('items', [])
            hex_color = color_to_hex(path.get('color'),
                                        path.get('stroke_opacity') or 1.0)
            width     = path.get('width') or 1.0

            # Special case: single 'qu' (Quad) — returned natively by
            # PyMuPDF >= 1.22 (Ubuntu 24.04+). Older versions resolve
            # 'qu' into 'l' segments automatically, so this is safe on
            # all versions.
            if len(items) == 1 and items[0][0] == 'qu':
                quad = items[0][1]
                corners = [
                    (quad.ul.x, quad.ul.y),
                    (quad.ur.x, quad.ur.y),
                    (quad.lr.x, quad.lr.y),
                    (quad.ll.x, quad.ll.y),
                    (quad.ul.x, quad.ul.y),   # close the shape
                ]
                make_stroke(layer_el, hex_color, width, corners,
                            [width] * len(corners))
                stroke_count += 1
                log(1, f"  [Geo P2a] quad stroke color={hex_color}")
                continue

            # Special case: single 're' → rect_to_strokes (correct corners)
            if len(items) == 1 and items[0][0] == 're':
                rect_to_strokes(layer_el, hex_color, width, items[0][1])
                stroke_count += 4
                continue

            for seg in path_to_pts(items):
                make_stroke(layer_el, hex_color, width, seg, [width] * len(seg))
                stroke_count += 1


        # ------------------------------------------------------------------
        # Pass 2b: rectangle-only fill paths → hline / vline / area
        # ------------------------------------------------------------------
        for path in rect_only_paths:
            if is_gray_fill(path.get('fill')):
                continue  # already rendered in Pass 1
            hex_color = color_to_hex(path['fill'], path.get('fill_opacity') or 1.0)
            for item in path.get('items', []):
                rect = item[1]
                kind = classify_rect_item(rect)

                if kind == 'corner':
                    continue
                elif kind == 'hline':
                    y_mid = (rect.y0 + rect.y1) / 2
                    make_stroke(layer_el, hex_color, LINE_WIDTH,
                                [(rect.x0, y_mid), (rect.x1, y_mid)])
                    stroke_count += 1
                elif kind == 'vline':
                    x_mid = (rect.x0 + rect.x1) / 2
                    make_stroke(layer_el, hex_color, LINE_WIDTH,
                                [(x_mid, rect.y0), (x_mid, rect.y1)])
                    stroke_count += 1
                elif kind == 'area':
                    fill_rect_solid(layer_el, hex_color, rect)
                    stroke_count += 1

                log(1, f"  [Geo P2] rect {kind} "
                       f"({rect.x0:.1f},{rect.y0:.1f})"
                       f"({rect.x1:.1f},{rect.y1:.1f})")

        # ------------------------------------------------------------------
        # Pass 2c: fill+stroke paths (type='fs') → filled area + outline
        # ------------------------------------------------------------------
        for path in filled_stroked_paths:
            items    = path.get('items', [])
            fill_hex = color_to_hex(path['fill'],   path.get('fill_opacity')   or 1.0)
            line_hex = color_to_hex(path['color'],  path.get('stroke_opacity') or 1.0)
            width    = path.get('width') or 1.0
            for seg in path_to_pts(items):
                make_filled_stroke(layer_el, fill_hex, seg)
                make_stroke(layer_el, line_hex, width, seg, [width] * len(seg))
                stroke_count += 2

        # ------------------------------------------------------------------
        # Pass 2d: fill-only curve paths (type='f') → filled shapes
        #
        # Donut detection via winding direction (Shoelace formula):
        # subpaths with opposite winding relative to the first subpath are
        # treated as holes and filled white.
        #
        # Limitation: Xournal++ has no clipping or even-odd fill support.
        # White holes may incorrectly cover later overlapping paths.
        # A warning is emitted for all compound paths (>= 2 subpaths).
        # ------------------------------------------------------------------
        for path in filled_curve_paths:
            items    = path.get('items', [])
            fill_hex = color_to_hex(path['fill'], path.get('fill_opacity') or 1.0)

            segs = path_to_pts(items)

            if len(segs) >= 2:
                log(1, f"  [Geo P2d] Warning: compound path with "
                       f"{len(segs)} subpaths – rendering in Xournal++ "
                       f"may be incomplete (even-odd/clipping not supported)")

            if len(segs) < 2:
                for seg in segs:
                    make_filled_stroke(layer_el, fill_hex, seg)
                    stroke_count += 1
                continue

            ref_sign = signed_area(segs[0])
            for seg in segs:
                sa = signed_area(seg)
                if ref_sign != 0.0 and sa != 0.0 and (sa > 0) != (ref_sign > 0):
                    # Opposite winding → inner subpath → approximate as white hole
                    make_filled_stroke(layer_el, "#ffffffff", seg)
                    log(2, f"  [Geo P2d] Donut hole: "
                           f"ref={ref_sign:.1f} seg={sa:.1f} → white")
                else:
                    make_filled_stroke(layer_el, fill_hex, seg)
                stroke_count += 1

    return stroke_count


def add_handwriting_filled(layer_el, handwriting):
    """
    Render handwriting paths as filled area strokes (default mode).
    Each path item ('l', 'c', 'qu', 're') is traced and emitted as a
    make_filled_stroke. Gaps between segments start a new stroke.
    """
    stroke_count = 0
    for path in handwriting:
        items     = path.get('items', [])
        hex_color = color_to_hex(path.get('fill'),
                                 path.get('fill_opacity') or 1.0)
        pts  = []
        last = None
        for item in items:
            op = item[0]
            if op == 'l':
                p0 = (item[1].x, item[1].y)
                p1 = (item[2].x, item[2].y)
                if last is None or pt_dist(p0, last) > EPS:
                    if pts:
                        make_filled_stroke(layer_el, hex_color, pts)
                        stroke_count += 1
                    pts = [p0, p1]
                else:
                    pts.append(p1)
                last = p1
            elif op == 'c':
                p0=(item[1].x,item[1].y); p1=(item[2].x,item[2].y)
                p2=(item[3].x,item[3].y); p3=(item[4].x,item[4].y)
                curve = bezier_pts(p0, p1, p2, p3)
                if last is None or pt_dist(p0, last) > EPS:
                    if pts:
                        make_filled_stroke(layer_el, hex_color, pts)
                        stroke_count += 1
                    pts = curve
                else:
                    pts.extend(curve[1:])
                last = p3
            elif op == 'qu':
                quad = item[1]
                p0 = (quad.ul.x, quad.ul.y)
                p1 = (quad.lr.x, quad.lr.y)
                if last is None or pt_dist(p0, last) > EPS:
                    if pts:
                        make_filled_stroke(layer_el, hex_color, pts)
                        stroke_count += 1
                    pts = [p0, p1]
                else:
                    pts.append(p1)
                last = p1
            elif op == 're':
                if pts:
                    make_filled_stroke(layer_el, hex_color, pts)
                    stroke_count += 1
                    pts = []
                rect_to_strokes(layer_el, hex_color, 0.0, item[1])
                stroke_count += 4
                last = None
        if pts:
            make_filled_stroke(layer_el, hex_color, pts)
            stroke_count += 1
    return stroke_count


def add_handwriting_line(layer_el, handwriting, ph):
    """
    Render handwriting paths as centerline strokes (--line mode).
    Each filled outline path is split into subpaths; the centroid of each
    subpath is used as a stroke point, and the average subpath radius
    determines the stroke width. This produces a single thin centerline
    stroke per ink outline instead of a filled area.
    """
    stroke_count = 0
    for path in handwriting:
        items     = path.get('items', [])
        hex_color = color_to_hex(path.get('fill'),
                                 path.get('fill_opacity') or 1.0)
        subpaths = split_into_subpaths(items, ph)
        if not subpaths: continue
        centers = []; radii = []
        for sp in subpaths:
            c = subpath_centroid(sp)
            if c is None: continue
            r = subpath_radius(sp, c)
            centers.append(c); radii.append(r)
        if not centers: continue
        avg_width = max(0.5, min(sum(radii)/len(radii)*2.0, 30.0))
        if len(centers) == 1:
            cx, cy = centers[0]; d = avg_width / 2
            pts = [(cx-d, cy), (cx+d, cy)]
        else:
            pts = centers
        make_stroke(layer_el, hex_color, avg_width, pts)
        stroke_count += 1
    return stroke_count


def add_highlighters(layer_el, highlighters):
    """
    Render semi-transparent fill paths as highlighter strokes.
    Paths with fill_opacity <= 0.9 are treated as highlighters.
    """
    stroke_count = 0
    for path in highlighters:
        items     = path.get('items', [])
        hex_color = color_to_hex(path.get('fill'), 1.0)
        pts  = []
        last = None
        for item in items:
            op = item[0]
            if op == 'l':
                p0 = (item[1].x, item[1].y)
                p1 = (item[2].x, item[2].y)
                if last is None or pt_dist(p0, last) > EPS:
                    if pts:
                        make_highlighter(layer_el, hex_color, pts)
                        stroke_count += 1
                    pts = [p0, p1]
                else:
                    pts.append(p1)
                last = p1
            elif op == 'c':
                p0=(item[1].x,item[1].y); p1=(item[2].x,item[2].y)
                p2=(item[3].x,item[3].y); p3=(item[4].x,item[4].y)
                curve = bezier_pts(p0, p1, p2, p3)
                if last is None or pt_dist(p0, last) > EPS:
                    if pts:
                        make_highlighter(layer_el, hex_color, pts)
                        stroke_count += 1
                    pts = curve
                else:
                    pts.extend(curve[1:])
                last = p3
        if pts:
            make_highlighter(layer_el, hex_color, pts)
            stroke_count += 1
    return stroke_count


def add_background_fill_lines(layer_el, bg_fill):
    """
    Render background fill paths as strokes:
      - Wide/tall rectangles → horizontal or vertical line strokes
      - Small filled circles (4 Bézier arcs, GoodNotes dot grid) → filled strokes
    """
    stroke_count = 0
    for path in bg_fill:
        col = path.get('fill')
        if col is None:
            continue
        hex_color = color_to_hex(col, path.get('fill_opacity') or 1.0)
        items = path.get('items', [])
        ops   = [it[0] for it in items]

        # GoodNotes dot grid: 4 Bézier arcs → render as small filled stroke
        if ops == ['c', 'c', 'c', 'c']:
            bbox = path.get('rect')
            if bbox is not None:
                segs = path_to_pts(items)
                for seg in segs:
                    if len(seg) >= 2:
                        make_filled_stroke(layer_el, hex_color, seg)
                        stroke_count += 1
            continue

        # Rectangles → horizontal or vertical line strokes
        for item in items:
            if item[0] == 're':
                r  = item[1]
                rw = abs(r.x1 - r.x0)
                rh = abs(r.y1 - r.y0)
                if rw > rh:
                    cy  = (r.y0 + r.y1) / 2
                    pts = [(r.x0, cy), (r.x1, cy)]
                    make_stroke(layer_el, hex_color, max(1.0, rh), pts)
                else:
                    cx  = (r.x0 + r.x1) / 2
                    pts = [(cx, r.y0), (cx, r.y1)]
                    make_stroke(layer_el, hex_color, max(1.0, rw), pts)
                stroke_count += 1

    return stroke_count


# ---------------------------------------------------------------------------
# Load existing .xopp
# ---------------------------------------------------------------------------

def load_xopp(xopp_path):
    """Load an existing .xopp file and return its root XML element."""
    with gzip.open(xopp_path, "rb") as f:
        data = f.read()
    return ET.fromstring(data)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_pdf_to_xopp(pdf_path, xopp_path, include_background=False,
                         force_line=False, include_ocr=False, append=False):
    """
    Convert a PDF file to a Xournal++ .xopp file.

    For each page:
      - Vector pages: extract drawings, classify into background/handwriting/
        geometry/highlighter/image/text, and render each category to the
        appropriate Xournal++ layer.
      - Raster pages (no vector paths, only images): render to bitmap,
        embed as background image layer, then skeletonize into stroke layer.

    If `append` is True and the target .xopp exists, pages are appended to it.
    """
    doc = fitz.open(pdf_path)
    print(f"Opening '{pdf_path}' → {len(doc)} page(s)")

    hw_mode = 'line' if force_line else 'filled'
    print(f"  Handwriting mode: {hw_mode}")
    if not include_background:
        print("  Background grid will be filtered out "
              "(use --include-background to disable)")
    if include_ocr:
        print("  OCR text will be embedded as invisible layer for search")

    if append and os.path.isfile(xopp_path):
        print(f"  Appending to existing '{xopp_path}'")
        root = load_xopp(xopp_path)
    else:
        print(f"  Creating new '{xopp_path}'")
        root = ET.Element("xournal")
        root.set("creator",     "pdf2xopp.py")
        root.set("fileversion", "4")
        ET.SubElement(root, "title").text = \
            "Xournal++ document - see https://xournalpp.github.io/"

    for page_num, page in enumerate(doc):
        pw = page.rect.width
        ph = page.rect.height
        print(f"\n  Page {page_num+1}: {pw:.1f} x {ph:.1f} pt")

        drawings = page.get_drawings()
        images   = page.get_images(full=True)

        page_el = ET.SubElement(root, "page")
        page_el.set("width",  f"{pw:.4f}")
        page_el.set("height", f"{ph:.4f}")
        bg = ET.SubElement(page_el, "background")
        bg.set("type",  "solid")
        bg.set("color", "#ffffffff")
        bg.set("style", "plain")

        # Raster page: no vector paths, only embedded images
        if len(drawings) == 0 and len(images) > 0:
            print("    No vector paths found – using raster strategy.")
            mat = fitz.Matrix(RESOLUTION / 72, RESOLUTION / 72)
            pix = page.get_pixmap(matrix=mat)
            make_image_layer(page_el, pw, ph, pix)
            make_skeleton_layer(page_el, page, pw, ph, pix)
            continue

        bg_stroke = [
            p for p in drawings
            if is_background_geometry(p, pw, ph)
            and p.get('color') is not None
            and p.get('fill') is None
        ]
        bg_fill = []
        for p in drawings:
            if p.get('fill') is None:
                continue
            if p.get('color') is not None:
                continue
            if p.get('fill') == WHITE:
                continue
            bg_result = is_background_fill(p, pw, ph)
            if bg_result == 'skip':
                pass              # Vollseiten-Rechteck: verwerfen
            elif bg_result:
                bg_fill.append(p)


        log(1, f"    bg_stroke: {len(bg_stroke)} paths")
        for p in bg_stroke[:5]:
            log(1, f"      color={p.get('color')}, "
                   f"opacity={p.get('stroke_opacity')}, "
                   f"items={len(p.get('items',[]))}, "
                   f"first_op={p.get('items',[['']])[0][0]}")
        log(1, f"    bg_fill: {len(bg_fill)} paths")

        if include_background:
            bg_layer_el = ET.SubElement(page_el, "layer")
            geo_bg  = add_geometry(bg_layer_el, bg_stroke, pw, ph)
            fill_bg = add_background_fill_lines(bg_layer_el, bg_fill)
            print(f"    Background layer: {geo_bg} geo + {fill_bg} fill stroke(s).")

        bg_set          = set(id(p) for p in bg_stroke + bg_fill)
        non_bg_drawings = [p for p in drawings if id(p) not in bg_set]

        # Main layer
        layer_el = ET.SubElement(page_el, "layer")

        # 1. Gray areas first (below everything else)
        geo_count  = add_geometry(layer_el, non_bg_drawings, pw, ph, pass_num=1)

        # 2. Embedded images
        add_images(layer_el, page, pw, ph, doc)

        # 3. Text (OCR layer inserted at bottom of page_el if include_ocr)
        added, suppressed = add_texts(layer_el, page, page_el, include_ocr)
        if added > 0 or suppressed > 0:
            print(f"    Textboxes: {added} added, {suppressed} suppressed (OCR).")

        filled_paths = [
            p for p in non_bg_drawings
            if p.get('fill') is not None
            and p.get('color') is None
            and p.get('fill') != WHITE
        ]
        handwriting  = [p for p in filled_paths
                        if (p.get('fill_opacity') or 1.0) > 0.9]
        highlighters = [p for p in filled_paths
                        if (p.get('fill_opacity') or 1.0) <= 0.9]

        # 4. Handwriting
        if hw_mode == 'filled':
            hw_count = add_handwriting_filled(layer_el, handwriting)
        else:
            hw_count = add_handwriting_line(layer_el, handwriting, ph)

        # 5. Highlighters
        hl_count = add_highlighters(layer_el, highlighters)

        # 6. Geometry (non-background, non-fill paths)
        geo_count += add_geometry(layer_el, non_bg_drawings, pw, ph, pass_num=2)

        print(f"    {len(drawings)} paths total – "
              f"{hw_count} handwriting ({hw_mode}), "
              f"{hl_count} highlighter, "
              f"{geo_count} geometry stroke(s).")

    ET.indent(root, space="  ")
    xml_bytes = (b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                 + ET.tostring(root, encoding="unicode").encode("utf-8"))
    with gzip.open(xopp_path, "wb") as f:
        f.write(xml_bytes)
    print(f"\nDone! '{xopp_path}' ({os.path.getsize(xopp_path)/1024:.1f} kB)")


def main():
    """
    Entry point. Parses command-line arguments, optionally converts SVG to PDF
    via Inkscape, then calls convert_pdf_to_xopp().
    """
    global LOGLEVEL
    parser = argparse.ArgumentParser(description="PDF/SVG → Xournal++ (.xopp)")
    parser.add_argument("pdf",
        metavar="input.pdf/svg",
        help="Input PDF or SVG file")
    parser.add_argument("--append", metavar="FILE.xopp",
        help="Append pages to existing .xopp file (created if missing)")
    parser.add_argument("--line", action="store_true",
        help="Use centerline (ellipse centroid) mode for handwriting")
    parser.add_argument("--include-background", action="store_true",
        help="Keep background grid lines as a separate layer")
    parser.add_argument("--include-ocr", action="store_true",
        help="Embed invisible OCR text (white) as bottom layer for search")
    parser.add_argument("--debug", type=int, default=0, metavar="N",
        help="Debug level (0=off, 1=normal, 2=verbose)")
    args = parser.parse_args()
    LOGLEVEL = args.debug

    if not os.path.isfile(args.pdf):
        print(f"Error: '{args.pdf}' not found.", file=sys.stderr)
        sys.exit(1)

    # SVG input: convert to PDF via Inkscape first
    pdf_path = args.pdf
    tmp_pdf  = None
    if args.pdf.lower().endswith('.svg'):
        inkscape = shutil.which('inkscape')
        if not inkscape:
            print("Error: SVG input requires Inkscape, but 'inkscape' was not found in PATH.",
                  file=sys.stderr)
            print("  Install with:  sudo apt install inkscape  (Debian/Ubuntu)", file=sys.stderr)
            print("                 sudo dnf install inkscape  (Fedora)", file=sys.stderr)
            print("                 brew install inkscape       (macOS)", file=sys.stderr)
            sys.exit(1)
        import tempfile, subprocess
        tmp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_pdf.close()
        pdf_path = tmp_pdf.name
        print(f"  Converting SVG → PDF via Inkscape ...")
        result = subprocess.run(
            [inkscape, '--export-type=pdf', args.pdf, '-o', pdf_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error: Inkscape conversion failed:\n{result.stderr}", file=sys.stderr)
            os.unlink(pdf_path)
            sys.exit(1)

    if args.append:
        xopp_path = args.append
    else:
        xopp_path = os.path.splitext(args.pdf)[0] + ".xopp"

    try:
        convert_pdf_to_xopp(pdf_path, xopp_path,
                            args.include_background,
                            args.line,
                            args.include_ocr,
                            append=bool(args.append))
    finally:
        if tmp_pdf is not None:
            os.unlink(pdf_path)


if __name__ == "__main__":
    main()
