"""
services/tools.py

Research tools for HormoneBench AI.

Compatible with:
- langchain 1.x
- langchain-core 1.x
- langchain-huggingface 1.x

Tools:
1. Local FAISS knowledge search
2. ArXiv academic paper search
3. Internet search using DDGS
"""

from __future__ import annotations

from typing import Optional

import arxiv
from ddgs import DDGS

from langchain.tools import BaseTool, tool

from services.vector_store import VectorStoreService


class ResearchTools:
    """
    Creates LangChain tools used by the research copilot.

    The tools are returned as BaseTool objects and can be passed
    directly to langchain.agents.create_agent().
    """

    def __init__(
        self,
        vector_store_service: Optional[VectorStoreService] = None,
        local_results: int = 4,
        arxiv_results: int = 3,
        web_results: int = 5,
    ) -> None:

        self.vector_store = (
            vector_store_service or VectorStoreService()
        )

        self.local_results = local_results
        self.arxiv_results = arxiv_results
        self.web_results = web_results

    # ==========================================================
    # Local FAISS tool
    # ==========================================================

    def _create_local_search_tool(self) -> BaseTool:
        """
        Create local vector-database search tool.
        """

        vector_store_service = self.vector_store
        result_limit = self.local_results

        @tool
        def search_local_knowledge(query: str) -> str:
            """
            Search HormoneBench local documents, uploaded research,
            benchmark documentation and clinical reference material.

            Use this tool before internet search when the question may
            be answered from local project documents.

            Args:
                query: Research question or search phrase.
            """

            if not query or not query.strip():
                return "Local search failed: query is empty."

            try:
                if not vector_store_service.vector_exists():
                    return (
                        "Local knowledge database is not available. "
                        "No FAISS vector database was found."
                    )

                vector_store = (
                    vector_store_service.load_vectorstore()
                )

                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": result_limit},
                )

                documents = retriever.invoke(query.strip())

                if not documents:
                    return (
                        "No relevant information was found "
                        "in the local knowledge database."
                    )

                formatted_results = []

                for index, document in enumerate(documents, start=1):

                    metadata = document.metadata or {}

                    source = (
                        metadata.get("source")
                        or metadata.get("file_name")
                        or metadata.get("filename")
                        or "Local document"
                    )

                    page = (
                        metadata.get("page")
                        or metadata.get("page_number")
                    )

                    content = " ".join(
                        document.page_content.split()
                    )

                    if len(content) > 2000:
                        content = content[:2000] + "..."

                    source_text = f"Source: {source}"

                    if page is not None:
                        source_text += f", Page: {page}"

                    formatted_results.append(
                        f"""
Local Result {index}
{source_text}

Content:
{content}
""".strip()
                    )

                return "\n\n".join(formatted_results)

            except Exception as error:
                return (
                    "Local knowledge search encountered an error: "
                    f"{type(error).__name__}: {error}"
                )

        return search_local_knowledge

    # ==========================================================
    # ArXiv tool
    # ==========================================================

    def _create_arxiv_search_tool(self) -> BaseTool:
        """
        Create academic-paper search tool.
        """

        result_limit = self.arxiv_results

        @tool
        def search_arxiv(query: str) -> str:
            """
            Search ArXiv for scientific papers and academic literature.

            Use this tool for academic evidence, machine-learning
            studies, biomedical research, hormonal-health research,
            PCOS studies and benchmark methodology.

            Args:
                query: Academic topic or paper search phrase.
            """

            if not query or not query.strip():
                return "ArXiv search failed: query is empty."

            try:
                client = arxiv.Client(
                    page_size=result_limit,
                    delay_seconds=3.0,
                    num_retries=2,
                )

                search = arxiv.Search(
                    query=query.strip(),
                    max_results=result_limit,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                papers = list(client.results(search))

                if not papers:
                    return (
                        "No relevant ArXiv papers were found "
                        "for this query."
                    )

                formatted_papers = []

                for index, paper in enumerate(papers, start=1):

                    authors = ", ".join(
                        author.name
                        for author in paper.authors[:6]
                    )

                    if len(paper.authors) > 6:
                        authors += ", et al."

                    summary = " ".join(
                        paper.summary.split()
                    )

                    if len(summary) > 1800:
                        summary = summary[:1800] + "..."

                    published = (
                        paper.published.date().isoformat()
                        if paper.published
                        else "Unknown"
                    )

                    formatted_papers.append(
                        f"""
ArXiv Paper {index}

Title: {paper.title}
Authors: {authors}
Published: {published}
Primary Category: {paper.primary_category}
Paper URL: {paper.entry_id}
PDF URL: {paper.pdf_url}

Abstract:
{summary}
""".strip()
                    )

                return "\n\n".join(formatted_papers)

            except Exception as error:
                return (
                    "ArXiv search encountered an error: "
                    f"{type(error).__name__}: {error}"
                )

        return search_arxiv

    # ==========================================================
    # Web-search tool
    # ==========================================================

    def _create_web_search_tool(self) -> BaseTool:
        """
        Create internet-search tool using DDGS.
        """

        result_limit = self.web_results

        @tool
        def search_web(query: str) -> str:
            """
            Search the public internet for recent information.

            Use this tool when local documents and ArXiv do not provide
            enough evidence, or when current information is required.
            Prefer trustworthy sources such as government, universities,
            peer-reviewed journals and established health organizations.

            Args:
                query: Web-search phrase.
            """

            if not query or not query.strip():
                return "Web search failed: query is empty."

            try:
                search_client = DDGS(timeout=12)

                results = search_client.text(
                    query=query.strip(),
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=result_limit,
                    backend="auto",
                )

                if not results:
                    return (
                        "No internet results were found "
                        "for this query."
                    )

                formatted_results = []

                for index, result in enumerate(results, start=1):

                    title = (
                        result.get("title")
                        or "Untitled result"
                    )

                    url = (
                        result.get("href")
                        or result.get("url")
                        or "URL unavailable"
                    )

                    body = (
                        result.get("body")
                        or result.get("snippet")
                        or "No description available."
                    )

                    body = " ".join(str(body).split())

                    if len(body) > 1200:
                        body = body[:1200] + "..."

                    formatted_results.append(
                        f"""
Web Result {index}

Title: {title}
URL: {url}

Summary:
{body}
""".strip()
                    )

                return "\n\n".join(formatted_results)

            except Exception as error:
                return (
                    "Internet search encountered an error: "
                    f"{type(error).__name__}: {error}"
                )

        return search_web

    # ==========================================================
    # Public method
    # ==========================================================

    def get_tools(self) -> list[BaseTool]:
        """
        Return all available research tools.

        Local search is only added when the FAISS database exists.
        ArXiv and web tools are always added.
        """

        available_tools: list[BaseTool] = []

        try:
            if self.vector_store.vector_exists():
                available_tools.append(
                    self._create_local_search_tool()
                )
        except Exception:
            # App must continue even if local FAISS is unavailable.
            pass

        available_tools.append(
            self._create_arxiv_search_tool()
        )

        available_tools.append(
            self._create_web_search_tool()
        )

        return available_tools


def get_research_tools(
    vector_store_service: Optional[
        VectorStoreService
    ] = None,
) -> list[BaseTool]:
    """
    Convenience function for creating all research tools.

    Example:
        tools = get_research_tools()
    """

    factory = ResearchTools(
        vector_store_service=vector_store_service
    )

    return factory.get_tools()