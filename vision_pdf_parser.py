"""
Vision-Based PDF Parser using Llama Vision (Local, Free)
Extracts all content from PDFs: tables, text, images, forms
Outputs: JSON, Markdown, HTML Report
"""

import json
import base64
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging
from dataclasses import dataclass, asdict
import requests

# PDF handling
try:
    from pdf2image import convert_from_path
except ImportError:
    print("Installing pdf2image...")
    os.system("pip install pdf2image --quiet")
    from pdf2image import convert_from_path

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system("pip install pdfplumber --quiet")
    import pdfplumber

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExtractedPage:
    """Represents one page of extracted content"""
    page_number: int
    raw_text: str
    tables: List[Dict]
    images_found: int
    vision_analysis: Dict
    timestamp: str

@dataclass
class ParsedDocument:
    """Complete parsed document"""
    filename: str
    total_pages: int
    extraction_timestamp: str
    pages: List[ExtractedPage]
    summary: Dict
    
    def to_dict(self):
        return {
            "filename": self.filename,
            "total_pages": self.total_pages,
            "extraction_timestamp": self.extraction_timestamp,
            "pages": [asdict(p) for p in self.pages],
            "summary": self.summary
        }

# ============================================================================
# VISION MODEL HANDLER (LLAMA VISION VIA OLLAMA)
# ============================================================================

class LlamaVisionHandler:
    """Handle Llama Vision API calls via Ollama"""
    
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "llava:latest"  # Llama Vision model
    
    @staticmethod
    def check_ollama():
        """Check if Ollama is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """Convert image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    @staticmethod
    def analyze_image_with_vision(image_path: str, prompt: str = None) -> Dict:
        """
        Analyze image using Llama Vision via Ollama
        Extracts: text, tables, structures, relationships
        """
        
        if prompt is None:
            prompt = """Analyze this document image comprehensively. Extract:
1. ALL visible text (preserve structure and order)
2. ALL tables (extract headers and data)
3. Any charts, diagrams, or visual elements
4. Forms, checkboxes, or structured fields
5. Page layout and document organization
6. Any numbers, codes, or special identifiers
7. Key information and relationships

Provide detailed, structured analysis. Be thorough."""
        
        try:
            # Read image and convert to base64
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode()
            
            # Call Ollama with vision model
            payload = {
                "model": "llava:latest",
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "analysis": result.get("response", ""),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "analysis": "",
                    "error": f"HTTP {response.status_code}"
                }
        
        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            return {
                "success": False,
                "analysis": "",
                "error": str(e)
            }

# ============================================================================
# PDF PARSER
# ============================================================================

class VisionPDFParser:
    """Parse PDFs using both traditional extraction and vision models"""
    
    def __init__(self, pdf_path: str, use_vision: bool = True):
        self.pdf_path = pdf_path
        self.use_vision = use_vision
        self.vision_handler = LlamaVisionHandler()
        self.filename = Path(pdf_path).name
        
        if use_vision and not self.vision_handler.check_ollama():
            logger.warning("⚠️ Ollama/Llama Vision not available. Using text extraction only.")
            self.use_vision = False
    
    def parse(self) -> ParsedDocument:
        """Parse entire PDF"""
        logger.info(f"Starting to parse: {self.filename}")
        
        pages_data = []
        vision_results = []
        table_summary = []
        
        # Get total pages
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        logger.info(f"Total pages: {total_pages}")
        
        # Process each page
        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}...")
            
            page_data = self._extract_page(page_num, total_pages)
            pages_data.append(page_data)
            
            if page_data.vision_analysis.get("success"):
                vision_results.append(page_data.vision_analysis.get("analysis", ""))
            
            if page_data.tables:
                table_summary.append({
                    "page": page_num + 1,
                    "table_count": len(page_data.tables)
                })
        
        # Create summary
        summary = {
            "total_pages_processed": total_pages,
            "total_tables_found": sum(len(p.tables) for p in pages_data),
            "total_images_analyzed": sum(p.images_found for p in pages_data),
            "extraction_method": "Vision + Text" if self.use_vision else "Text Only",
            "pages_with_tables": len(table_summary),
            "vision_enabled": self.use_vision
        }
        
        parsed_doc = ParsedDocument(
            filename=self.filename,
            total_pages=total_pages,
            extraction_timestamp=datetime.now().isoformat(),
            pages=pages_data,
            summary=summary
        )
        
        logger.info("✅ PDF parsing completed")
        return parsed_doc
    
    def _extract_page(self, page_num: int, total_pages: int) -> ExtractedPage:
        """Extract single page using both traditional and vision methods"""
        
        # Traditional extraction (fast, reliable for text)
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_num]
            
            # Extract text
            raw_text = page.extract_text() or ""
            
            # Extract tables
            tables = []
            page_tables = page.extract_tables()
            if page_tables:
                for table_idx, table in enumerate(page_tables):
                    tables.append({
                        "table_id": f"table_{page_num}_{table_idx}",
                        "page": page_num + 1,
                        "data": table,
                        "formatted": self._format_table(table)
                    })
        
        # Vision analysis (comprehensive, slower)
        vision_result = {"success": False, "analysis": "", "model": "llama_vision"}
        
        if self.use_vision:
            # Convert page to image
            images = convert_from_path(
                self.pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=150
            )
            
            if images:
                # Save image temporarily
                temp_image_path = f"/tmp/page_{page_num}.png"
                images[0].save(temp_image_path)
                
                # Analyze with vision
                vision_analysis = self.vision_handler.analyze_image_with_vision(
                    temp_image_path,
                    prompt=self._get_vision_prompt(page_num, total_pages)
                )
                
                vision_result = vision_analysis
                
                # Clean up
                os.remove(temp_image_path)
        
        extracted_page = ExtractedPage(
            page_number=page_num + 1,
            raw_text=raw_text[:2000],  # Limit to 2000 chars for storage
            tables=tables,
            images_found=1 if self.use_vision else 0,
            vision_analysis=vision_result,
            timestamp=datetime.now().isoformat()
        )
        
        return extracted_page
    
    @staticmethod
    def _format_table(table: List[List]) -> str:
        """Format table as readable text"""
        if not table:
            return ""
        
        headers = table[0]
        result = ["| " + " | ".join(str(h) for h in headers) + " |"]
        result.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for row in table[1:]:
            result.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(result)
    
    @staticmethod
    def _get_vision_prompt(page_num: int, total_pages: int) -> str:
        """Get specialized prompt based on page context"""
        base_prompt = """Analyze this page thoroughly and extract:
1. ALL TEXT (preserve formatting, structure, order)
2. ALL TABLES (with headers and complete data)
3. ANY VISUAL ELEMENTS (diagrams, charts, layouts)
4. KEY INFORMATION (numbers, dates, names, codes)
5. DOCUMENT STRUCTURE (sections, hierarchy)
6. IMPORTANT RELATIONSHIPS (between data elements)

Be comprehensive and accurate. Format findings clearly."""
        
        return base_prompt

# ============================================================================
# OUTPUT GENERATORS
# ============================================================================

class OutputGenerator:
    """Generate output files in multiple formats"""
    
    def __init__(self, output_dir: str = "./results/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def save_json(self, parsed_doc: ParsedDocument) -> str:
        """Save as JSON (structured, queryable)"""
        output_file = self.output_dir / f"{Path(parsed_doc.filename).stem}_parsed.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_doc.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON saved: {output_file}")
        return str(output_file)
    
    def save_markdown(self, parsed_doc: ParsedDocument) -> str:
        """Save as Markdown (readable, portable)"""
        output_file = self.output_dir / f"{Path(parsed_doc.filename).stem}_parsed.md"
        
        md_content = self._generate_markdown(parsed_doc)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        logger.info(f"✅ Markdown saved: {output_file}")
        return str(output_file)
    
    def save_html_report(self, parsed_doc: ParsedDocument) -> str:
        """Save as interactive HTML report"""
        output_file = self.output_dir / f"{Path(parsed_doc.filename).stem}_report.html"
        
        html_content = self._generate_html_report(parsed_doc)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"✅ HTML Report saved: {output_file}")
        return str(output_file)
    
    @staticmethod
    def _generate_markdown(parsed_doc: ParsedDocument) -> str:
        """Generate markdown content"""
        md = f"""# Document Parse Report: {parsed_doc.filename}

**Extraction Timestamp:** {parsed_doc.extraction_timestamp}
**Total Pages:** {parsed_doc.total_pages}
**Extraction Method:** {parsed_doc.summary['extraction_method']}

## Summary
- Total Pages Processed: {parsed_doc.summary['total_pages_processed']}
- Total Tables Found: {parsed_doc.summary['total_tables_found']}
- Total Images Analyzed: {parsed_doc.summary['total_images_analyzed']}
- Vision Enabled: {parsed_doc.summary['vision_enabled']}

---

"""
        
        # Add page-by-page content
        for page in parsed_doc.pages:
            md += f"## Page {page.page_number}\n\n"
            
            # Add extracted text
            if page.raw_text:
                md += "### Extracted Text\n```\n"
                md += page.raw_text[:1000] + "...\n" if len(page.raw_text) > 1000 else page.raw_text
                md += "\n```\n\n"
            
            # Add tables
            if page.tables:
                md += f"### Tables ({len(page.tables)})\n\n"
                for table in page.tables:
                    md += f"#### {table['table_id']}\n\n"
                    md += table['formatted'] + "\n\n"
            
            # Add vision analysis
            if page.vision_analysis.get("success"):
                md += "### Vision Analysis\n\n"
                analysis = page.vision_analysis.get("analysis", "")
                md += analysis[:2000] + "...\n\n" if len(analysis) > 2000 else analysis + "\n\n"
            
            md += "---\n\n"
        
        return md
    
    @staticmethod
    def _generate_html_report(parsed_doc: ParsedDocument) -> str:
        """Generate interactive HTML report"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Parse Report - {parsed_doc.filename}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header .meta {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-card .number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .page {{
            margin-bottom: 40px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .page-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 2px solid #e9ecef;
            font-weight: bold;
            font-size: 1.2em;
            color: #667eea;
        }}
        
        .page-content {{
            padding: 20px;
        }}
        
        .section {{
            margin-bottom: 25px;
        }}
        
        .section-title {{
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #667eea;
        }}
        
        .text-content {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            max-height: 400px;
            overflow-y: auto;
            line-height: 1.5;
            color: #333;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        table th {{
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }}
        
        table td {{
            padding: 10px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        table tr:hover {{
            background: #f8f9fa;
        }}
        
        .vision-content {{
            background: #e7f3ff;
            padding: 15px;
            border-left: 4px solid #2196F3;
            border-radius: 5px;
            margin-top: 10px;
        }}
        
        .no-data {{
            color: #999;
            font-style: italic;
            padding: 10px;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-right: 5px;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📄 PDF Parse Report</h1>
            <div class="meta">
                <p><strong>{parsed_doc.filename}</strong></p>
                <p>Extracted: {parsed_doc.extraction_timestamp}</p>
            </div>
        </header>
        
        <div class="summary">
            <div class="stat-card">
                <div class="number">{parsed_doc.summary['total_pages_processed']}</div>
                <div class="label">Pages Processed</div>
            </div>
            <div class="stat-card">
                <div class="number">{parsed_doc.summary['total_tables_found']}</div>
                <div class="label">Tables Found</div>
            </div>
            <div class="stat-card">
                <div class="number">{parsed_doc.summary['total_images_analyzed']}</div>
                <div class="label">Images Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="label">{parsed_doc.summary['extraction_method']}</div>
            </div>
        </div>
        
        <div class="content">
"""
        
        # Add page-by-page content
        for page in parsed_doc.pages:
            html += f"""
            <div class="page">
                <div class="page-header">Page {page.page_number}</div>
                <div class="page-content">
"""
            
            # Text content
            if page.raw_text:
                html += f"""
                    <div class="section">
                        <div class="section-title">📝 Extracted Text</div>
                        <div class="text-content">{page.raw_text[:1000]}{'...' if len(page.raw_text) > 1000 else ''}</div>
                    </div>
"""
            
            # Tables
            if page.tables:
                html += f"""
                    <div class="section">
                        <div class="section-title">📊 Tables ({len(page.tables)})</div>
"""
                for table in page.tables:
                    html += f"""
                        <div style="margin-top: 15px;">
                            <strong>{table['table_id']}</strong>
                            <table>
                                <thead>
                                    <tr>
                                        {"".join([f"<th>{h}</th>" for h in table['data'][0]])}
                                    </tr>
                                </thead>
                                <tbody>
                                    {"".join([f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>" for row in table['data'][1:]])}
                                </tbody>
                            </table>
                        </div>
"""
                html += "</div>"
            
            # Vision analysis
            if page.vision_analysis.get("success"):
                analysis = page.vision_analysis.get("analysis", "")
                html += f"""
                    <div class="section">
                        <div class="section-title">🤖 Vision Analysis</div>
                        <div class="vision-content">{analysis[:2000]}{'...' if len(analysis) > 2000 else ''}</div>
                    </div>
"""
            
            html += """
                </div>
            </div>
"""
        
        html += """
        </div>
        
        <footer>
            <p>Generated by Vision PDF Parser | Powered by Llama Vision</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         Vision-Based PDF Parser (Llama Vision Local)           ║
    ║              Extract Everything from PDFs                      ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if PDF path provided
    if len(sys.argv) < 2:
        print("\n❌ Usage: python vision_pdf_parser.py <pdf_file_path>")
        print("\nExample:")
        print("  python vision_pdf_parser.py HR_Manual.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Validate PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Parse PDF
    parser = VisionPDFParser(pdf_path, use_vision=True)
    parsed_doc = parser.parse()
    
    # Generate outputs
    output_gen = OutputGenerator("./results/")
    
    json_file = output_gen.save_json(parsed_doc)
    md_file = output_gen.save_markdown(parsed_doc)
    html_file = output_gen.save_html_report(parsed_doc)
    
    # Display summary
    print("\n" + "="*70)
    print("✅ PARSING COMPLETE")
    print("="*70)
    print(f"\n📄 Document: {parsed_doc.filename}")
    print(f"📊 Pages Processed: {parsed_doc.summary['total_pages_processed']}")
    print(f"📋 Tables Found: {parsed_doc.summary['total_tables_found']}")
    print(f"🖼️  Images Analyzed: {parsed_doc.summary['total_images_analyzed']}")
    print(f"🔧 Extraction Method: {parsed_doc.summary['extraction_method']}")
    
    print("\n📁 Output Files Generated:")
    print(f"  1. JSON: {json_file}")
    print(f"  2. Markdown: {md_file}")
    print(f"  3. HTML Report: {html_file}")
    
    print(f"\n🌐 View HTML Report:")
    print(f"  Open in browser: file://{Path(html_file).absolute()}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
