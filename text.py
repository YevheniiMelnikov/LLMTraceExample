INPUT_TEXT = """
Knowledge Base Preparation. GaussMaster aggregates multisource GaussDB documents into a unified knowledge base, splitting text by semantic boundaries (e.g., the code block including
GaussDB SQLs), retaining structural elements (e.g., the hierarchical
titles in markdown files), and removing duplicates. Each chunk is
augmented with version tags and neighboring context. Additionally,
GaussMaster wraps DBMind’s 25 diagnostic tools under a RESTful
interface and codifies expert workflows into anomaly diagnosis
trees, improving the troubleshooting stability of LLMs.
LLM-based Question Pre-processing. When a user raises a natural language question, GaussMaster first performs two steps of
pre-processing: (1) Hybrid Question Decomposition: the question
is decomposed into finer-grained sub-queries (e.g., smaller semantic chunks or tool-invocation hints) via both rule-based and LLM
methods. (2) Intent-Aware Prompt Assignment: Based on the question’s content and intent, GaussMaster automatically routes by
prompting LLMs the question into either the LLM-based Q&A module (including the code copilot functionality like code review and
optimization) or the LLM-based Diagnosis & Repair module (if it
detects an anomaly-related or troubleshooting intent).
LLM-based Q&A. We provide three main functions: (1) Risk checking mechanism with both a sensitive word detector and a semanticbased content classifier. Once the preprocessed questions are considered to be risky, GaussMaster returns a pre-defined refusal response to users; (2) Hybrid knowledge retrieval for matching the
most relevant information (from sources like the development guide
and the FAQ textbook) to answer the question. We fine-tune an
embedding model and a re-ranking model over the GaussDB corpus
with 106,810 samples by expanding the input question set through
rule-based synonym substitution and LLM-based question rephrasing. The retrieved information is utilized to generate high-quality
answers; (3) The generated answer goes through the same risk
checking mechanism to ensure safety before responding to users.
LLM-based Diagnosis & Repair. We call the module given the
description of triggered alerts, which supports four main functions:
(1) Identify the correct tools through hybrid retrieval and re-ranking
based on user question and tool usage descriptions; (2) Multi-step
strategy to fill in the corresponding parameters by analyzing user
questions and asking for the complement of absent ones from users;
(3) Retrieve the relevant historical alarm cases and diagnosis tree
(i.e., the sequence of troubleshooting steps by GaussDB experts) for
the subsequent diagnosis; (4) Conduct diagnosis-tree-guided orchestration using multiple tools, combined with a multi-agent diagnosis
approach that includes expert assignment, task decomposition, and
self-reflection [12].
"""

REFERENCE = (
    "GaussMaster aggregates structured GaussDB documentation into a searchable knowledge base, "
    "adds versioning and context, and wraps diagnostic tools into workflows. "
    "It preprocesses user questions by decomposing them and routing based on intent. "
    "The Q&A system includes risk detection, hybrid retrieval, and fine-tuned models for answer generation. "
    "The diagnosis module uses multi-step retrieval, parameter completion, and orchestrated tool usage with multi-agent reasoning."
)