import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "sanskrit_chroma_db_fresh"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3.8-27b"
TOP_K = 8

# Retrieval gate. This is deliberately conservative: the model may use
# its own grammar knowledge ONLY when retrieval finds no relevant context.
DENSE_DISTANCE_THRESHOLD = 1.25
COMMON_ENGLISH_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}

TECHNICAL_TERM_ALIASES = {
    "kṣaipra": [
        "kṣaipra", "ksaipra", "kshaipra", "क्षैप्र", "क्षैप्रः",
    ],
    "praśliṣṭa": [
        "praśliṣṭa", "prashlista", "prashlishta", "प्रश्लिष्ट", "प्रश्लिष्टः",
    ],
    "abhinihita": [
        "abhinihita", "अभिनिहित", "अभिनिहितः",
    ],
    "jātyasvarita": [
        "jātyasvarita", "jatyasvarita", "जात्यस्वरित", "जात्यस्वरितः",
    ],
}

SECTION_RELATIONSHIPS = {
    "जात्योत्पत्तिः": ["क्षैप्रः", "प्रश्लिष्टः", "अभिनिहितः"],
}
SECTION_PARENTS = {
    child: parent
    for parent, children in SECTION_RELATIONSHIPS.items()
    for child in children
}

# ============================================================
# BM25
# ============================================================
def normalize_for_lexical_search(text):
    return unicodedata.normalize("NFC", text).casefold()


def tokenize_for_lexical_search(text):
    text = normalize_for_lexical_search(text)
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_documents = [
            tokenize_for_lexical_search(doc.page_content) for doc in documents
        ]
        self.doc_lengths = [len(x) for x in self.tokenized_documents]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0
        )
        self.document_frequency = defaultdict(int)
        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequency[token] += 1
        self.total_documents = len(documents)

    def idf(self, token):
        df = self.document_frequency.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(
            1 + (self.total_documents - df + 0.5) / (df + 0.5)
        )

    def score_document(self, query_tokens, document_index):
        document_tokens = self.tokenized_documents[document_index]
        if not document_tokens or not self.avg_doc_length:
            return 0.0

        tf_counts = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0

        for token in query_tokens:
            tf = tf_counts.get(token, 0)
            if tf == 0:
                continue
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (document_length / self.avg_doc_length)
            )
            score += self.idf(token) * numerator / denominator

        return score

    def search(self, query, k=20):
        query_tokens = tokenize_for_lexical_search(query)
        if not query_tokens:
            return []

        scored = []
        for index in range(self.total_documents):
            score = self.score_document(query_tokens, index)
            if score > 0:
                scored.append((self.documents[index], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ============================================================
# STRENGTHENED HYBRID RETRIEVER
# ============================================================
def expand_technical_query(query):
    normalized = unicodedata.normalize("NFC", query).casefold()
    additions = []

    for aliases in TECHNICAL_TERM_ALIASES.values():
        for alias in sorted(aliases, key=len, reverse=True):
            if unicodedata.normalize("NFC", alias).casefold() in normalized:
                additions.extend(aliases)
                break

    if not additions:
        return query

    return query + " " + " ".join(dict.fromkeys(additions))


class HybridRetriever:
    def __init__(self, vectorstore, documents, dense_weight=1.0, lexical_weight=1.5, rrf_k=60):
        self.vectorstore = vectorstore
        self.documents = documents
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(documents)
        self.section_to_doc = {
            doc.metadata.get("section", ""): doc for doc in documents
        }

    @staticmethod
    def document_identity(doc):
        return (
            doc.metadata.get("source", ""),
            doc.metadata.get("section", ""),
        )

    def dense_search(self, query, k=20):
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def lexical_search(self, query, k=20):
        return self.bm25.search(expand_technical_query(query), k=k)

    def _expand_relationships(self, fused_scores, document_lookup):
        sections_seen = {
            document_lookup[doc_id].metadata.get("section", "")
            for doc_id in fused_scores
            if doc_id in document_lookup
        }

        parents_to_expand = set()
        for section in sections_seen:
            if section in SECTION_RELATIONSHIPS:
                parents_to_expand.add(section)
            if section in SECTION_PARENTS:
                parents_to_expand.add(SECTION_PARENTS[section])

        for parent in parents_to_expand:
            parent_doc = self.section_to_doc.get(parent)
            if parent_doc is None:
                continue

            parent_id = self.document_identity(parent_doc)
            parent_score = fused_scores.get(parent_id, 0.0)
            if parent_score <= 0:
                continue

            for child_section in SECTION_RELATIONSHIPS[parent]:
                child_doc = self.section_to_doc.get(child_section)
                if child_doc is None:
                    continue

                child_id = self.document_identity(child_doc)
                document_lookup[child_id] = child_doc
                relationship_score = parent_score * 0.90
                fused_scores[child_id] = max(
                    fused_scores.get(child_id, 0.0), relationship_score
                )

    def search(self, query, k=8):
        dense_results = self.dense_search(query, max(20, k))
        lexical_results = self.lexical_search(query, max(20, k))

        fused_scores = defaultdict(float)
        document_lookup = {}

        for rank, (doc, _distance) in enumerate(dense_results, start=1):
            doc_id = self.document_identity(doc)
            document_lookup[doc_id] = doc
            fused_scores[doc_id] += self.dense_weight / (self.rrf_k + rank)

        for rank, (doc, _score) in enumerate(lexical_results, start=1):
            doc_id = self.document_identity(doc)
            document_lookup[doc_id] = doc
            fused_scores[doc_id] += self.lexical_weight / (self.rrf_k + rank)

        self._expand_relationships(fused_scores, document_lookup)
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        results = [
            (document_lookup[doc_id], score)
            for doc_id, score in ranked[:k]
        ]

        return results, dense_results, lexical_results

    def has_relevant_context(self, query, results, dense_results, lexical_results):
        """Conservative retrieval gate.

        Relevant context is considered present when either:
        1. lexical retrieval found a real corpus match, or
        2. dense retrieval is sufficiently close to the query.

        The LLM is never given permission to use outside knowledge when this
        gate says that relevant corpus context exists.
        """
        query_tokens = set(tokenize_for_lexical_search(expand_technical_query(query)))
        meaningful_query_tokens = {
            token
            for token in query_tokens
            if token not in COMMON_ENGLISH_TOKENS and len(token) > 1
        }

        if meaningful_query_tokens and lexical_results:
            top_doc_tokens = set(
                tokenize_for_lexical_search(lexical_results[0][0].page_content)
            )
            if meaningful_query_tokens & top_doc_tokens:
                return True

        if dense_results:
            best_distance = dense_results[0][1]
            if best_distance <= DENSE_DISTANCE_THRESHOLD:
                return True

        return False


# ============================================================
# LOAD DATABASE ONCE
# ============================================================
@st.cache_resource(show_spinner="Loading Sanskrit grammar database...")
def load_retriever():
    if not DB_DIR.exists():
        raise FileNotFoundError(
            "Sanskrit grammar database is missing. "
            "Please ensure the 'sanskrit_chroma_db_fresh' folder is deployed "
            "alongside app.py."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    raw = vectorstore.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(
            raw.get("documents", []),
            raw.get("metadatas", []),
        )
    ]

    if not documents:
        raise RuntimeError("The Sanskrit grammar database contains no documents.")

    return HybridRetriever(vectorstore, documents)


# ============================================================
# CONTEXT + GROQ
# ============================================================
def build_context(results):
    blocks = []

    for doc, _score in results:
        section = doc.metadata.get("section", "")
        blocks.append(
            f"Section: {section}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = """You are a Sanskrit grammar tutor.

You have access to a retrieval database containing Sanskrit grammar material.

PRIMARY RULE — RETRIEVED KNOWLEDGE:
When relevant retrieved context is supplied, answer using that context as the
primary and authoritative basis. Do not replace it with general knowledge.
Preserve the source's terminology, distinctions, Sanskrit forms, examples,
and stated scope. Do not invent rules, examples, translations, historical
claims, or explanations that contradict or exceed the retrieved material.

NO-RETRIEVAL FALLBACK:
Only when the system explicitly says that NO relevant database context was
found may you answer from your own general knowledge of Sanskrit grammar.
When doing so, make it clear that the answer comes from general knowledge
rather than the tutor's retrieved material. Do not pretend that general
knowledge came from the database.

IMPORTANT:
- Never mention the retrieval system, database, chunks, rankings, sources,
  source numbers, R1/R2, metadata, or internal implementation.
- Never say "Source 1", "Source 2", "according to the retrieved source",
  or similar internal-reference language in the student-facing answer.
- If relevant database context is supplied but it does not establish a
  requested detail, say that the available material does not establish that
  detail. Do not silently fill that gap from general knowledge.
- Explain Sanskrit technical terms clearly when useful.
- Preserve Sanskrit forms accurately.
- Be concise but sufficiently explanatory for a student.
"""


def generate_answer(query, context, groq_api_key, context_found):
    if context_found:
        retrieval_instruction = """RELEVANT DATABASE CONTEXT WAS FOUND.
You MUST answer from the supplied context. Do not use outside knowledge to
fill missing details. If the context does not establish part of the answer,
say so explicitly."""
        context_block = context
    else:
        retrieval_instruction = """NO RELEVANT DATABASE CONTEXT WAS FOUND.
You may answer from your general knowledge of Sanskrit grammar. Clearly
indicate that this part of the answer is based on general knowledge rather
than the tutor's database."""
        context_block = "(No relevant database context was found.)"

    user_prompt = f"""{retrieval_instruction}

Question:
{query}

Retrieved context:
{context_block}

Give a clear, helpful tutor-style answer."""

    if not groq_api_key:
        raise ValueError(
            "The Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "reasoning_effort": "none",
        "max_completion_tokens": 1400,
    }

    response = requests.post(
        f"{GROQ_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# ============================================================
# PUBLIC STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Sanskrit Grammar Tutor",
    page_icon="ॐ",
    layout="centered",
)

st.title("ॐ Sanskrit Grammar Tutor")
st.caption("Ask questions about Sanskrit grammar and Vedic accents.")

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("The tutor is not configured yet. Please contact the site administrator.")
    st.stop()

try:
    retriever = load_retriever()
except Exception:
    st.error("The Sanskrit grammar database could not be loaded. Please try again later.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a Sanskrit grammar question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                results, dense_results, lexical_results = retriever.search(
                    query,
                    k=TOP_K,
                )
                context_found = retriever.has_relevant_context(
                    query,
                    results,
                    dense_results,
                    lexical_results,
                )
                context = build_context(results) if context_found else ""
                answer = generate_answer(
                    query,
                    context,
                    groq_api_key,
                    context_found,
                )
            except requests.exceptions.Timeout:
                answer = "The tutor took too long to respond. Please try again."
            except requests.exceptions.ConnectionError:
                answer = "The tutor service could not be reached. Please try again shortly."
            except requests.exceptions.HTTPError:
                answer = "The tutor service returned an error. Please try again shortly."
            except Exception:
                answer = "Something went wrong while answering your question. Please try again."

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
