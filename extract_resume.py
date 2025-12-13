
import pypdf

try:
    reader = pypdf.PdfReader("Rishabh Tole Resume.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open("resume_text.txt", "w") as f:
        f.write(text)
    print("Successfully extracted text to resume_text.txt")
except Exception as e:
    print(f"Error extracting text: {e}")
