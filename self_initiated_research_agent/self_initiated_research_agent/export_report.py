import os
import logging
import glob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Configure logging
logging.basicConfig(filename="agent.log", level=logging.INFO)
logging.info("Export report script started")

# Find the most recent markdown report
reports_dir = os.path.join(os.path.dirname(__file__), "reports")
md_files = glob.glob(os.path.join(reports_dir, "report_*.md"))
if not md_files:
    logging.error("No markdown reports found.")
    print("No markdown reports found.")
    exit()

# Get the most recent report
latest_md = max(md_files, key=os.path.getctime)
md_path = latest_md

# Generate output paths based on the markdown filename
base_name = os.path.splitext(os.path.basename(md_path))[0]
pdf_path = os.path.join(reports_dir, f"{base_name}.pdf")
html_path = os.path.join(reports_dir, f"{base_name}.html")

logging.info(f"Using markdown file: {md_path}")

if not os.path.exists(md_path):
    logging.error("report.md not found.")
    print("report.md not found.")
    exit()

logging.info(f"Reading markdown file: {md_path}")
# Read the Markdown file as plain text
with open(md_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

logging.info("Creating PDF from markdown content")
# Create a simple PDF with the Markdown content as plain text
c = canvas.Canvas(pdf_path, pagesize=letter)
y = 750
for line in lines:
    c.drawString(50, y, line.strip())
    y -= 15
    if y < 50:
        c.showPage()
        y = 750
c.save()
logging.info(f"PDF saved successfully: {pdf_path}")
print(f"PDF saved: {pdf_path}")

logging.info("Creating HTML from markdown content")
# Save the Markdown as a simple HTML file (no formatting)
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><body>\n")
    for line in lines:
        f.write(f"<p>{line.strip()}</p>\n")
    f.write("</body></html>\n")
logging.info(f"HTML saved successfully: {html_path}")
print(f"HTML saved: {html_path}")

logging.info("Export report script completed successfully")