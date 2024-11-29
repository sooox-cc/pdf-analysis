# Advanced PDF Analysis (Proof Of Concept)

A PDF analysis tool built with Streamlit that provides document analysis including text extraction, sentiment analysis, keyword identification, and various visualizations.

## Features

- **Text Extraction & Structure Analysis**: Extract and analyze document structure, including headers and content blocks
- **Sentiment Analysis**: Analyze sentiment patterns throughout the document
- **Keyword Extraction**: Identify key terms and phrases using TF-IDF
- **Word Frequency Analysis**: Visualize word usage patterns
- **Document Statistics**: Generate comprehensive document metrics
- **Interactive Visualizations**: Multiple interactive plots for data exploration
- **Document Preview**: Quick access to document content and structure

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a PDF file using the file uploader

4. View the analysis results including:
   - Document metadata
   - Text structure analysis
   - Keyword extraction
   - Word frequency visualizations
   - Sentiment analysis
   - Document preview

## Technical Details

The application uses several key technologies:
- **Streamlit**: For the web interface
- **PyMuPDF**: For PDF processing
- **Transformers**: For sentiment analysis using DistilBERT
- **NLTK**: For text processing
- **Plotly**: For interactive visualizations
- **Pandas & NumPy**: For data manipulation
- **scikit-learn**: For TF-IDF feature extraction

## Error Handling

The application includes comprehensive error handling for:
- PDF file processing
- Text extraction
- Model loading
- Memory management
- Invalid file formats
