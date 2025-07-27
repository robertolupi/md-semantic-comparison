# Markdown Semantic Comparison

This project provides an interactive web-based tool to perform a deep semantic comparison of two Markdown or text documents.

The application, built with Streamlit, goes beyond simple text diffing. It leverages modern AI models to identify and categorize underlying concepts and themes within the documents. It then uses sentence embeddings to visually highlight where these concepts appear in each text, offering a nuanced, side-by-side view of their similarities and differences.

## Features

-   **Interactive UI**: A user-friendly web interface for uploading and comparing files.
-   **Side-by-Side View**: Displays both documents concurrently, with synchronized highlighting.
-   **AI-Powered Concept Analysis**: Uses a configurable AI model (via Google, OpenAI, or a local provider) to determine which themes are unique to each document and which are shared.
-   **Semantic Sentence Highlighting**: Color-codes sentences in each document based on their semantic similarity to the identified concepts.
-   **Configurable AI Providers**: Flexibility to choose between different AI models from providers like Google and OpenAI, or even a local model.
-   **Adjustable Similarity Threshold**: A slider to fine-tune the sensitivity of the sentence-to-concept matching.
-   **Markdown Frontmatter Support**: Automatically extracts and displays YAML frontmatter from Markdown files.

## How It Works

The comparison process involves two main stages:

1.  **Concept Extraction**: First, the contents of both documents are sent to a selected Large Language Model (LLM). The LLM analyzes the texts and returns a structured list of key concepts, categorized into three groups:
    -   Concepts present only in the first document.
    -   Concepts present only in the second document.
    -   Concepts present in both documents.

2.  **Sentence-Level Highlighting**: Next, the application uses a `sentence-transformers` model to generate numerical vector embeddings for both the extracted concepts and each sentence in the documents. It calculates the cosine similarity between each sentence and the list of concepts. If a sentence's meaning is close enough to a concept (exceeding a user-defined threshold), it is highlighted with a unique color corresponding to that concept.

This two-step approach provides both a high-level thematic summary and a detailed, in-context visualization of where those themes manifest.

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd md-semantic-comparison
    ```

2.  **Install the required Python packages:**
    ```bash
    uv pip install .
    ```
    This command reads the `pyproject.toml` file and installs all necessary dependencies.

3.  **Download NLTK data:**
    The application uses the NLTK library for sentence tokenization. You may need to download the required data package. Run the `install_runtime_deps.py` script to do this automatically:
    ```bash
    python install_runtime_deps.py
    ```

## Usage

To run the web application, execute the following command in your terminal:

```bash
streamlit run compare.py
```

This will start a local web server and open the application in your default browser.

From there, you can:
1.  Select an AI Provider (e.g., Google, OpenAI, or a local endpoint).
2.  Choose a specific model from the dropdown.
3.  Upload your two documents (`.md` or `.txt`) using the file uploaders in the sidebar.
4.  Adjust the similarity threshold if desired.
5.  Click the **"Run Analysis"** button to start the comparison.
