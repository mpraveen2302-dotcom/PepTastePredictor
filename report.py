from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_report(metrics):
    c = canvas.Canvas("analytics_report.pdf", pagesize=A4)
    y = 800
    for k, v in metrics.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20
    c.save()
    return "analytics_report.pdf"
