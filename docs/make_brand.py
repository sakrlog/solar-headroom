"""Draw the integration's brand images (custom_components/solar_headroom/brand/).

A sun whose lower part is filled and upper part left open: the solar power in use, and the headroom
above it. Needs Pillow (in Home Assistant's environment):  python docs/make_brand.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "custom_components" / "solar_headroom" / "brand"
AMBER = (249, 168, 37, 255)
SS = 4                                   # supersampling


def draw(size: int, color=AMBER) -> Image.Image:
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r = s / 2, s * 0.25                # sun disk
    ring = s * 0.045
    # rays
    for i in range(12):
        a = math.radians(i * 30 + 15)
        r0, r1 = r + s * 0.085, r + s * 0.20
        x0, y0 = c + r0 * math.cos(a), c + r0 * math.sin(a)
        x1, y1 = c + r1 * math.cos(a), c + r1 * math.sin(a)
        w = s * 0.05
        d.line([(x0, y0), (x1, y1)], fill=color, width=round(w))
        for x, y in ((x0, y0), (x1, y1)):
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=color)
    # open ring (the potential)
    d.ellipse([c - r, c - r, c + r, c + r], outline=color, width=round(ring))
    # filled lower part (what is used), with a straight level line
    level = c - r * 0.1
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([c - r, c - r, c + r, c + r], fill=255)
    md.rectangle([0, 0, s, level], fill=0)
    img.paste(Image.new("RGBA", (s, s), color), (0, 0), mask)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, size in (("icon.png", 256), ("icon@2x.png", 512)):
        draw(size).save(OUT / name, optimize=True)
        print(OUT / name)


if __name__ == "__main__":
    main()
