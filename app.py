import fitz
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from collections import Counter
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AdvancedPDFAnalyzer:
    def __init__(self):
        try:
            # Initialize sentiment analysis model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            # Basic stopwords set
            self.stop_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 
                                 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 
                                 'as', 'you', 'do', 'at'])
            
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            raise

    def extract_structured_content(self, doc):
        """Extract structured content including headers and sections"""
        structure = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                item = {
                                    "page": page_num + 1,
                                    "text": text,
                                    "font_size": span["size"],
                                    "font": span["font"],
                                    "type": "header" if span["size"] > 11 else "content"
                                }
                                structure.append(item)
        return structure

    def extract_keywords(self, text, top_n=10):
        """Extract key terms using TF-IDF"""
        try:
            # Preprocess text
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Create document matrix
            tfidf_matrix = self.vectorizer.fit_transform([text])
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Create keyword dictionary
            keywords = [
                {"word": word, "score": score} 
                for word, score in zip(feature_names, scores)
            ]
            
            # Sort by score and take top_n
            keywords.sort(key=lambda x: x["score"], reverse=True)
            return keywords[:top_n]
            
        except Exception as e:
            st.error(f"Error extracting keywords: {str(e)}")
            return []

    def create_word_frequency_plot(self, text):
        """Create word frequency visualization"""
        try:
            # Simple word tokenization
            words = text.lower().split()
            # Remove punctuation and numbers
            words = [w for w in words if w.isalpha() and len(w) > 2]
            # Remove stopwords
            words = [w for w in words if w not in self.stop_words]
            
            # Calculate frequency
            word_freq = Counter(words).most_common(20)
            
            # Create DataFrame for plotting
            df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
            
            # Create plot
            fig = px.bar(
                df,
                x='word',
                y='frequency',
                title='Top 20 Most Frequent Words',
                labels={'word': 'Word', 'frequency': 'Frequency'},
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating word frequency plot: {str(e)}")
            return None

    def create_font_size_distribution(self, structure):
        """Create visualization of font size distribution"""
        try:
            sizes = [item["font_size"] for item in structure]
            
            fig = px.histogram(
                x=sizes,
                nbins=20,
                title='Distribution of Font Sizes',
                labels={'x': 'Font Size', 'y': 'Count'},
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating font size distribution: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        """Analyze sentiment using PyTorch"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            score = predictions.numpy()[0]
            label = "POSITIVE" if score[1] > score[0] else "NEGATIVE"
            
            return {"label": label, "score": float(max(score))}
            
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "ERROR", "score": 0.0}

    def analyze_paragraphs(self, paragraphs):
        """Analyze sentiment for each paragraph"""
        results = []
        for p in paragraphs:
            if len(p.split()) > 3:
                sentiment = self.analyze_sentiment(p)
                results.append({
                    'text': p[:100] + '...',
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                })
        return results

    def extract_text_from_pdf(self, pdf_path):
        """Extract text and metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            metadata = {
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown"),
                "pages": len(doc),
                "format": doc.metadata.get("format", "Unknown"),
                "creation_date": doc.metadata.get("creationDate", "Unknown")
            }
            
            # Extract structured content
            structure = self.extract_structured_content(doc)
            
            # Extract text
            for item in structure:
                text += item["text"] + " "
            
            return text.strip(), metadata, structure
            
        except Exception as e:
            st.error(f"Error extracting PDF content: {str(e)}")
            return "", {}, []

    def generate_advanced_analytics(self, text, structure):
        """Generate comprehensive text analytics"""
        try:
            # Basic word tokenization
            words = text.lower().split()
            # Simple sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            # Calculate statistics
            analytics = {
                'total_words': len(words),
                'unique_words': len(set(words)),
                'avg_word_length': np.mean([len(w) for w in words]),
                'total_sentences': len(sentences),
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences]),
                'headers_count': len([i for i in structure if i['type'] == 'header']),
                'content_blocks': len([i for i in structure if i['type'] == 'content'])
            }
            
            return analytics
            
        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")
            return {}

def main():
    st.title("Advanced PDF Analysis Dashboard")
    st.markdown("""
    This dashboard provides detailed analysis of PDF documents including:
    - Text extraction and structure analysis
    - Sentiment analysis
    - Keyword extraction
    - Word frequency analysis
    - Document statistics
    """)
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            pdf_path = Path("temp.pdf")
            pdf_path.write_bytes(uploaded_file.getvalue())
            
            # Initialize analyzer
            analyzer = AdvancedPDFAnalyzer()
            
            with st.spinner('Analyzing PDF...'):
                # Extract and analyze text
                text, metadata, structure = analyzer.extract_text_from_pdf(pdf_path)
                
                if text:
                    # Document Overview
                    st.header("Document Overview")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Metadata")
                        st.write(metadata)
                    
                    with col2:
                        st.subheader("Structure")
                        st.write(f"Headers: {len([i for i in structure if i['type'] == 'header'])}")
                        st.write(f"Content Blocks: {len([i for i in structure if i['type'] == 'content'])}")
                    
                    # Text Analysis
                    st.header("Text Analysis")
                    analytics = analyzer.generate_advanced_analytics(text, structure)
                    st.write(analytics)
                    
                    # Keyword Analysis
                    st.header("Keyword Analysis")
                    keywords = analyzer.extract_keywords(text)
                    st.write(pd.DataFrame(keywords))
                    
                    # Visualizations
                    st.header("Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        word_freq_fig = analyzer.create_word_frequency_plot(text)
                        if word_freq_fig:
                            st.plotly_chart(word_freq_fig)
                    
                    with col2:
                        font_dist_fig = analyzer.create_font_size_distribution(structure)
                        if font_dist_fig:
                            st.plotly_chart(font_dist_fig)
                    
                    # Sentiment Analysis
                    st.header("Sentiment Analysis")
                    paragraphs = [i["text"] for i in structure if i["type"] == "content"]
                    sentiment_results = analyzer.analyze_paragraphs(paragraphs)
                    
                    if sentiment_results:
                        df = pd.DataFrame(sentiment_results)
                        fig = px.bar(
                            df,
                            x='text',
                            y='score',
                            color='sentiment',
                            title='Sentiment Analysis by Paragraph'
                        )
                        st.plotly_chart(fig)
                    
                    # Document Preview
                    st.header("Document Preview")
                    st.subheader("First 500 characters")
                    st.text_area("Preview", text[:500], height=200)
                    
                    # Document Structure
                    st.header("Document Structure")
                    structure_df = pd.DataFrame(structure)
                    st.dataframe(structure_df)
            
            # Cleanup
            pdf_path.unlink()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()