from crewai.tools import BaseTool
from pypdf import PdfReader


class CustomPdfReader(BaseTool):
    name: str = "Pdf Reader"
    description: str = (
        "Read Pdf files."
    )

    def _run(self, file_path: str) -> str:
        reader = PdfReader(file_path)

        pdf_text = ""

        # Loop through each page in the PDF
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()  # Extract text from the page

        return pdf_text
