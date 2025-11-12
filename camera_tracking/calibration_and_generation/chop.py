#!/usr/bin/env python3
"""
Take generated plate_layout.pdf (300mm diameter plate) and produce
a 2-page PDF suitable for standard printer paper (Letter/A4),
splitting the plate in half for true-size printing.
"""
import fitz
import os

output_dir = "generated"

input_pdf = os.path.join(output_dir, "/Users/brendanchharawala/Documents/GitHub/ball-balancer/camera_tracking/calibration_and_generation/generated/plate_layout.pdf")
output_pdf = os.path.join(output_dir, "/Users/brendanchharawala/Documents/GitHub/ball-balancer/camera_tracking/calibration_and_generation/generated/plate_2page_print.pdf")

# Open the original PDF
doc = fitz.open(input_pdf)
page = doc[0]

# Original page dimensions in points
w, h = page.rect.width, page.rect.height

# Create new PDF
out_doc = fitz.open()

# --- Left half ---
left_page = out_doc.new_page(width=w/2, height=h)
left_page.show_pdf_page(left_page.rect, doc, 0, clip=fitz.Rect(0, 0, w/2, h))

# --- Right half ---
right_page = out_doc.new_page(width=w/2, height=h)
right_page.show_pdf_page(right_page.rect, doc, 0, clip=fitz.Rect(w/2, 0, w, h))

out_doc.save(output_pdf)
print("2-page printable PDF saved:", output_pdf)




