# PDF Analysis PoC

A proof-of-concept application that demonstrates PDF text extraction, AI-powered analysis, and interactive visualization capabilities. This project uses modern NLP techniques to analyze PDF documents and present insights through a Streamlit dashboard.

## Features

- PDF text and metadata extraction
- Text preprocessing and analysis
- Sentiment analysis using transformer models
- Interactive visualizations of analysis results
- Web-based dashboard interface

## Requirements

- Python 3.8+
- Dependencies:
  ```
  PyMuPDF
  pandas
  numpy
  transformers
  streamlit
  plotly
  ```

## Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a PDF file using the file uploader

4. View the analysis results in the dashboard:
   - Document metadata
   - Text analytics
   - Sentiment analysis visualization
   - Extracted text preview

## Project Structure

```
pdf-analysis-poc/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Future Enhancements

- Table extraction from PDFs
- OCR support for scanned documents
- Document classification
- Named Entity Recognition
- Topic modeling
- Database integration for result storage
- Batch processing capabilities