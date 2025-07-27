import asyncio
import re
import typing
from collections import OrderedDict
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import streamlit as st
import yaml
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelName as GoogleModels
from pydantic_ai.models.openai import OpenAIModel, AllModels as OpenAIModels
from pydantic_ai.providers.openai import OpenAIProvider
from sentence_transformers import SentenceTransformer, util


def build_palette(size: int, cmap_name: str) -> list[str]:
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(0.2 + 0.8 * i / max(size - 1, 1))) for i in range(size)]


def build_palette_map(lists: list[list[str]], cmaps: list[str]) -> OrderedDict[str, str]:
    palette = OrderedDict()
    for lst, cmap in zip(lists, cmaps):
        p = build_palette(len(lst), cmap)
        for i, col in zip(lst, p):
            palette[i] = col
    return palette

def extract_sentences_from_markdown(markdown_content):
    """Tokenizes markdown content into sentences, preserving markdown syntax."""
    sentences = sent_tokenize(markdown_content)
    return [sent.strip() for sent in sentences if sent.strip()]

def colorize_concepts(md: str, palette: dict[str, str], normalize: bool = True,
                      threshold : float = 0.6) -> str:
    concepts = list(palette.keys())
    sentences = extract_sentences_from_markdown(md)
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_sentences = st_model.encode(sentences, convert_to_tensor=True, normalize_embeddings=normalize)
    emb_concepts = st_model.encode(concepts, convert_to_tensor=True, normalize_embeddings=normalize)

    for i, s in enumerate(sentences):
        sims = util.cos_sim(emb_sentences[i], emb_concepts)[0]
        # Compute the highest score above threshold
        highest_score = 0
        closest_concept = None
        for j, score in enumerate(sims):
            if score > highest_score and score > threshold:
                highest_score = score
                closest_concept = concepts[j]
        if closest_concept:
            md = md.replace(s, f'<span style="color: {palette[closest_concept]}">({concepts.index(closest_concept)}) {s}</span>')
    return md

class ComparisonResult(BaseModel):
    in_both: list[str]
    only_in_first: list[str]
    only_in_second: list[str]


def get_literals_from_type(type_hint: typing.Any) -> list[str]:
    """
    Extracts all string literals from a type hint that may contain
    typing.Literal, including those nested in a typing.Union.
    """
    literals = []

    # Get the arguments of the top-level type (e.g., the members of the Union)
    # For a Union, this will be (str, Literal[...], Literal[...])
    union_args = typing.get_args(type_hint)

    # Iterate through each type within the Union
    for arg in union_args:
        # Check if the origin of the type is Literal
        if typing.get_origin(arg) is typing.Literal:
            # If it is, get its arguments, which are the actual string values
            literal_values = typing.get_args(arg)
            literals.extend(literal_values)

    return literals


FRONTMATTER_RE = re.compile(r'---\n(.*?)\n---\n', re.DOTALL | re.MULTILINE)


def get_file_content(md_file) -> tuple[dict[str, Any], str]:
    content = md_file.read().decode('utf-8')
    if m := FRONTMATTER_RE.search(content):
        fm = yaml.safe_load(m.group(1))
        return fm, FRONTMATTER_RE.sub('', content)
    return {}, content


# Page Configuration

st.set_page_config(layout="wide")
st.title("Semantic Comparison of Markdown files")

with st.sidebar:
    st.title("Settings")
    provider_name = st.selectbox("Provider", ["local", "Google", "OpenAI"])
    model = None
    match provider_name:
        case "local":
            base_url = st.text_input("Base URL", "http://localhost:1234/v1")
            model_name = st.text_input("Model Name")
            model = OpenAIModel(model_name=model_name, provider=OpenAIProvider(base_url=base_url))
        case "Google":
            model_name = st.selectbox("Model Name", get_literals_from_type(GoogleModels))
            model = GoogleModel(model_name=model_name)
        case "OpenAI":
            model_name = st.selectbox("Model Name", get_literals_from_type(OpenAIModels))
            model = OpenAIModel(model_name=model_name)

    file1 = st.file_uploader("First File", type=["md", "txt"])
    file2 = st.file_uploader("Second File", type=["md", "txt"])

    fm1, doc1 = get_file_content(file1) if file1 else ({}, None)
    fm2, doc2 = get_file_content(file2) if file2 else ({}, None)

    threshold = st.slider('Threshold', 0.0, 1.0, 0.5, 0.01)

    if model:
        agent = Agent(model=model,
                      output_type=ComparisonResult,
                      system_prompt="Read the two documents and identify themes and concepts; "
                                    "then list those that are in both documents, "
                                    "those that are only in the first document, "
                                    "those that are only in the second document")

    if doc1 and doc2 and model:
        do_analysis = st.button("Run Analysis")
    else:
        do_analysis = None

if do_analysis:
    with st.spinner("Running analysis..."):
        agent_result = asyncio.run(
            agent.run("---First document---\n{doc1}\n\n---Second document---\n{doc2}".format(doc1=doc1, doc2=doc2)))
        st.session_state["results"] = agent_result.output

if "results" in st.session_state:
    results = st.session_state["results"]

    palette = build_palette_map([results.in_both, results.only_in_first, results.only_in_second],
                                ["Greens", "Reds", "Blues"])
    palette_list = list(palette.keys())
    if results:
        col1, common, col2 = st.columns(3)
        with col1:
            st.header("Only in first")
            for c in results.only_in_first:
                st.html(f"<div style='color: {palette[c]}'>({palette_list.index(c)}) {c}</div>")
        with common:
            st.header("In both")
            for c in results.in_both:
                st.html(f"<div style='color: {palette[c]}'>({palette_list.index(c)}) {c}</div>")
        with col2:
            st.header("Only in second")
            for c in results.only_in_second:
                st.html(f"<div style='color: {palette[c]}'>({palette_list.index(c)}) {c}</div>")
                
    with st.spinner("Tagging concepts..."):
        doc1 = colorize_concepts(doc1, palette, threshold=threshold)
        doc2 = colorize_concepts(doc2, palette, threshold=threshold)

col1, col2 = st.columns(2)

with col1:
    if file1:
        st.header(file1.name)
        if fm1:
            exp1 = st.expander("Frontmatter")
            exp1.write(fm1)
        st.markdown(doc1, unsafe_allow_html=True)

with col2:
    if file2:
        st.header(file2.name)
        if fm2:
            exp2 = st.expander("Frontmatter")
            exp2.write(fm2)
        st.markdown(doc2, unsafe_allow_html=True)
