import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr


load_dotenv()

# Load and prepare data
import os

# Get the directory where the script itself is located
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "books_with_emotions.csv")
books = pd.read_csv(file_path)
books["large_thumbnail"] = books["thumbnail"].apply(
    lambda x: x + "&fife=w800" if pd.notna(x) else os.path.join(base_path, "52107_original.jpg")
)


# Load and process documents
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "tagged_description.txt")
raw_documents = TextLoader(file_path, encoding="utf-8").load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)

# Create vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    
    # If no query, return random books
    if not query or query.strip() == "":
        book_recs = books.sample(n=min(final_top_k, len(books)))
    else:
        # Semantic search
        recs = db_books.similarity_search_with_score(
            query,
            k=initial_top_k
        )

        # Extract ISBNs from retrieved documents
        books_list = []

        for rec, _ in recs:
            first_token = rec.page_content.strip().split()[0]
            
            # Nettoyage : enlever guillemets et caractères parasites
            cleaned = first_token.replace('"', '').replace("'", "")
            
            if cleaned.isdigit():
                books_list.append(int(cleaned))

        # Filter dataframe
        book_recs = books[books["isbn13"].isin(books_list)]

    # Filter by category
    if category and category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]

    # Sort by tone
    if tone and tone != "All":
        if tone == "Happy":
            book_recs = book_recs.sort_values(by="joy", ascending=False)
        elif tone == "Surprising":
            book_recs = book_recs.sort_values(by="surprise", ascending=False)
        elif tone == "Sad":
            book_recs = book_recs.sort_values(by="sadness", ascending=False)
        elif tone == "Angry":
            book_recs = book_recs.sort_values(by="anger", ascending=False)
        elif tone == "Suspenseful":
            book_recs = book_recs.sort_values(by="fear", ascending=False)

    # Return top results
    return book_recs.head(final_top_k)


def recommend_books(query: str, category: str, tone: str):
    """Generate book recommendations based on user input"""
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = str(row["description"]) if pd.notna(row["description"]) else "No description available."
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors = str(row["authors"]) if pd.notna(row["authors"]) else "Unknown Author"
        authors_split = authors.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors

        caption = f"**{row['title']}**\n*{authors_str}*\n\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


def load_initial_books():
    """Load random books when the app starts"""
    return recommend_books("", "All", "All")


categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom CSS for professional beige design
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

body {
    background: linear-gradient(135deg, #f5e6d3 0%, #e8d5c4 50%, #d4c4b0 100%) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: transparent !important;
}

#main-container {
    background: linear-gradient(135deg, #faf6f1 0%, #f0e8dc 100%) !important;
    border-radius: 24px !important;
    padding: 50px 40px !important;
    box-shadow: 0 20px 60px rgba(101, 67, 33, 0.15) !important;
    border: 1px solid rgba(139, 101, 66, 0.1) !important;
}

.header-title {
    font-family: 'Playfair Display', serif !important;
    color: #5d4037 !important;
    font-size: 56px !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 15px !important;
    text-shadow: 2px 2px 4px rgba(93, 64, 55, 0.1) !important;
    letter-spacing: -1px !important;
    background: linear-gradient(135deg, #6d4c41 0%, #8d6e63 50%, #a1887f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.header-subtitle {
    color: #8d6e63 !important;
    font-size: 19px !important;
    text-align: center !important;
    margin-bottom: 45px !important;
    font-weight: 400 !important;
    font-style: italic !important;
}

.input-section {
    background: rgba(255, 248, 240, 0.7) !important;
    border-radius: 18px !important;
    padding: 35px !important;
    margin-bottom: 35px !important;
    border: 2px solid rgba(141, 110, 99, 0.15) !important;
    box-shadow: 0 8px 24px rgba(93, 64, 55, 0.08) !important;
}

label {
    color: #5d4037 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

input, select, textarea {
    background: #ffffff !important;
    border: 2px solid rgba(141, 110, 99, 0.25) !important;
    border-radius: 12px !important;
    color: #3e2723 !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
}

input:focus, select:focus, textarea:focus {
    border-color: #8d6e63 !important;
    box-shadow: 0 0 0 4px rgba(141, 110, 99, 0.15) !important;
    outline: none !important;
    background: #fffbf5 !important;
}

button {
    background: linear-gradient(135deg, #8d6e63 0%, #6d4c41 100%) !important;
    color: #fff8f0 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 6px 20px rgba(93, 64, 55, 0.25) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(93, 64, 55, 0.35) !important;
    background: linear-gradient(135deg, #6d4c41 0%, #5d4037 100%) !important;
}

.gallery {
    background: rgba(255, 248, 240, 0.5) !important;
    border-radius: 18px !important;
    padding: 25px !important;
    border: 2px solid rgba(141, 110, 99, 0.15) !important;
}

.gallery img {
    border-radius: 12px !important;
    box-shadow: 0 10px 30px rgba(93, 64, 55, 0.2) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    border: 3px solid rgba(141, 110, 99, 0.1) !important;
}

.gallery img:hover {
    transform: scale(1.05) translateY(-5px) !important;
    box-shadow: 0 15px 40px rgba(93, 64, 55, 0.3) !important;
}

.output-label {
    font-family: 'Playfair Display', serif !important;
    color: #5d4037 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin-bottom: 20px !important;
}

::placeholder {
    color: #a1887f !important;
    opacity: 0.7 !important;
    font-style: italic !important;
}

.dropdown-arrow {
    color: #5d4037 !important;
}

/* Gallery caption styling */
.gallery .caption {
    background: linear-gradient(to bottom, transparent 0%, rgba(93, 64, 55, 0.9) 100%) !important;
    color: #fff8f0 !important;
}

/* FORCE BLACK TITLES - Multiple selectors for maximum specificity */
.gallery p,
.gallery .caption-label p,
div.gallery p {
    color: #000000 !important;
    line-height: 1.7 !important;
    margin: 5px 0 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

.gallery strong,
.gallery .caption-label strong,
div.gallery strong,
.gallery p strong {
    color: #000000 !important;
    font-size: 20px !important;
    font-weight: 900 !important;
    display: block !important;
    margin-bottom: 8px !important;
    text-shadow: 0 1px 2px rgba(255,255,255,0.5) !important;
}

.gallery em,
.gallery .caption-label em,
div.gallery em,
.gallery p em {
    color: #2a2a2a !important;
    display: block !important;
    margin-bottom: 10px !important;
    font-weight: 600 !important;
}

/* Override any inherited text colors */
.gallery * {
    color: inherit !important;
}

.gallery .caption-label {
    color: #000000 !important;
}
"""

# Launch the app
if __name__ == "__main__":
   
    from pyngrok import ngrok, conf
    import os
    
    ngrok_token = "3983aXBXPFg4fwkC0QUzqT1iccZ_aqSeu6Cv5iZmvbuhkK5A"  
    
    if ngrok_token != "3983aXBXPFg4fwkC0QUzqT1iccZ_aqSeu6Cv5iZmvbuhkK5A":
        conf.get_default().auth_token = ngrok_token
        
        # Créer le tunnel ngrok
        public_url = ngrok.connect(7860)
        print("\n" + "="*60)
        print(f"🌐 URL PUBLIQUE : {public_url}")
        print("="*60 + "\n")
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        with gr.Column(elem_id="main-container"):
            gr.HTML('<h1 class="header-title">📚 Literary Discovery</h1>')
            gr.HTML('<p class="header-subtitle">Uncover your next great read through intelligent recommendations</p>')
            
            with gr.Column(elem_classes="input-section"):
                query = gr.Textbox(
                    label="Describe Your Perfect Book",
                    placeholder="e.g., A gripping psychological thriller with unexpected twists...",
                    lines=2,
                    value=""
                )
                
                with gr.Row():
                    category = gr.Dropdown(
                        categories,
                        label="Genre",
                        value="All"
                    )
                    tone = gr.Dropdown(
                        tones,
                        label="Emotional Tone",
                        value="All"
                    )
                
                btn = gr.Button("✨ Discover Books", size="lg")
            
            output = gr.Gallery(
                label="Your Personalized Recommendations",
                columns=4,
                rows=2,
                height="auto",
                object_fit="cover",
                elem_classes="gallery",
                value=load_initial_books()
            )

            btn.click(
                fn=recommend_books,
                inputs=[query, category, tone],
                outputs=output
            )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True
    )