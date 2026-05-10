"""Generate the MNEME token image. Minimalist substrate-identity motif:
concentric rings (slow weights / substrate layers), central Greek mu (Μ)
for Mneme — Greek goddess of memory.

Output: assets/mneme.png (512x512 PNG, <1MB per Clanker constraints).
"""

from PIL import Image, ImageDraw, ImageFont
import os, math

W, H = 512, 512
BG = (10, 10, 16)  # near-black with hint of indigo (substrate-night)
CX, CY = W // 2, H // 2

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img, "RGBA")

# Subtle radial gradient — slightly lighter at center to suggest "lit substrate"
for r in range(W // 2, 0, -2):
    alpha = int(8 * (1 - r / (W / 2)) ** 2)  # very subtle
    draw.ellipse([CX - r, CY - r, CX + r, CY + r], fill=(255, 255, 255, alpha))

# Concentric rings — substrate layers, slow weights
# Each ring slightly thinner/dimmer than the previous; suggests depth & sleep cycles
rings = [
    (210, 1, (240, 240, 245, 60)),   # outermost — faint
    (180, 1, (240, 240, 245, 90)),
    (150, 1, (240, 240, 245, 130)),
    (120, 2, (240, 240, 245, 170)),
    (90,  2, (240, 240, 245, 210)),  # innermost — brightest
]
for r, w, color in rings:
    draw.ellipse([CX - r, CY - r, CX + r, CY + r], outline=color, width=w)

# Central glyph: Greek capital mu (Μ) — clean, recognizable, on-theme.
# Use a system font; fall back gracefully.
def find_font(size):
    candidates = [
        "C:/Windows/Fonts/seguisb.ttf",      # Segoe UI Semibold
        "C:/Windows/Fonts/segoeui.ttf",      # Segoe UI
        "C:/Windows/Fonts/arialbd.ttf",      # Arial Bold
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()

# Greek mu Μ
glyph = "Μ"
font = find_font(180)
bbox = draw.textbbox((0, 0), glyph, font=font)
gw = bbox[2] - bbox[0]
gh = bbox[3] - bbox[1]
# Pillow textbbox includes ascender/descender padding; recenter manually.
tx = CX - gw // 2 - bbox[0]
ty = CY - gh // 2 - bbox[1]
# Soft shadow for depth
draw.text((tx + 3, ty + 3), glyph, fill=(0, 0, 0, 120), font=font)
draw.text((tx, ty), glyph, fill=(245, 245, 250, 255), font=font)

# Wordmark below: MNEME
font_word = find_font(28)
word = "MNEME"
bbox_w = draw.textbbox((0, 0), word, font=font_word)
ww = bbox_w[2] - bbox_w[0]
wh = bbox_w[3] - bbox_w[1]
wx = CX - ww // 2 - bbox_w[0]
wy = CY + 130 - bbox_w[1]
draw.text((wx, wy), word, fill=(180, 180, 195, 255), font=font_word)

# Subtitle — micro, very subtle
sub = "substrate identity"
font_sub = find_font(14)
bbox_s = draw.textbbox((0, 0), sub, font=font_sub)
sw = bbox_s[2] - bbox_s[0]
sx = CX - sw // 2 - bbox_s[0]
sy = wy + wh + 12
draw.text((sx, sy), sub, fill=(120, 120, 135, 255), font=font_sub)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mneme.png")
img.save(out, format="PNG", optimize=True)
size = os.path.getsize(out)
print(f"Wrote {out}  size={size} bytes ({size/1024:.1f} KB)")
