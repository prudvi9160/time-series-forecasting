import sys
import pandas as pd
import re

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
    import pdfplumber

pdf_path = r"C:\Users\prudv\Downloads\Forecasting Case- Study.xlsx - Sheet1.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    
    lines = all_text.strip().split('\n')
    print(f"Extracted {len(lines)} lines from PDF")
    print("First 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i}: {repr(line[:80])}")
    
    # Try to parse lines that look like data rows
    # Pattern: State Name, Date (M/D/YYYY), Amount (with commas/parens), Category
    rows = []
    state_pattern = re.compile(r'^([A-Z][a-z\s]+?)\s+(\d{1,2}/\d{1,2}/\d{4})\s+[\(\d,\)]+\s+(Beverages|[A-Za-z\s]+)$')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 20:
            continue
        
        # Try simple parsing: look for date pattern
        if re.search(r'\d{1,2}/\d{1,2}/\d{4}', line):
            print(f"Potential data line: {repr(line[:100])}")
            # Try to extract components
            # State is at start, Date is M/D/YYYY, Amount is in parens or number, Category is at end
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
