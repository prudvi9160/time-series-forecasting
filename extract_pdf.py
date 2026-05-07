import sys
import pandas as pd

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
        print(f"PDF has {len(pdf.pages)} pages")
        
        all_tables = []
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                print(f"Page {page_num + 1}: Found {len(tables)} table(s)")
                for table_idx, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df)
                        print(f"  Table {table_idx}: {len(table)} rows, {len(table[0])} columns")
                        print(f"    Columns: {table[0]}")
                        print(f"    First row: {table[1] if len(table) > 1 else 'N/A'}")
        
        if all_tables:
            combined_df = pd.concat(all_tables, ignore_index=True)
            output_path = r"c:\Users\prudv\OneDrive\Desktop\quickhyreai\data\raw.xlsx"
            combined_df.to_excel(output_path, index=False)
            print(f"\nSaved {len(combined_df)} rows to {output_path}")
        else:
            print("No tables found in PDF")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
