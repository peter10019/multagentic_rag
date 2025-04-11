import wikipedia
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import tool, ReactCodeAgent, Tool

from multirag.cfg import RetrievalCfg, SearchCfg
from multirag.engine import ModelEngine
from multirag.utils import format_logs


class PageRetriever:
    def __init__(self,
                 llm_engine: ModelEngine,
                 embeddings: HuggingFaceEmbeddings,
                 system_prompt: str,
                 retrieval_cfg: RetrievalCfg = RetrievalCfg()
                 ):
        self.embeddings = embeddings
        self.retrieval_cfg = retrieval_cfg
        self.passage_search_agent = ReactCodeAgent(system_prompt=system_prompt,
                                                   llm_engine=llm_engine,
                                                   # Reduce maximum amount of tries to retrieve information from the same page
                                                   max_iterations=retrieval_cfg.max_iterations,
                                                   tools=[self.passages_retrieval_tool()])
        self.page = None
        self.vector_db = None
        self.logs = []

    def compute_page_embeddings(self) -> None:
        """Compute the embeddings of the Wikipedia page content and create a vector database for passage retrieval."""
        content = self.page.content
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.retrieval_cfg.chunk_size,
                                                  chunk_overlap=self.retrieval_cfg.chunk_overlap,
                                                  separators=self.retrieval_cfg.separators)
        page_splits = splitter.create_documents(texts=[content])
        self.vector_db = FAISS.from_documents(page_splits, self.embeddings)

    def passages_retrieval_tool(self) -> Tool:
        @tool
        def retrieve_passages(query: str) -> str:
            """Use this tool to retrieve passages relevant to the given query from the Wikipedia page. Always print and inspect the output of this tool, do not parse its output using regex.
                If you already used this tool for a query, and you didn't get the information you needed, try calling it again with a more specific query. If the page doesn't seem to contain the information you needed, return the final answer stating that the Wikipedia page don't contain the necessary information to answer the query.  Don't call this tool twice with the same query.
                Args:
                    query: The query to search for.
                Returns:
                    str: The passages retrieved.
            """
            passages = self.vector_db.similarity_search(query, k=self.retrieval_cfg.n_passages)
            caption = f"""Retrieved passages for query "{query}":\n"""
            return caption + ' ...\n'.join(
                [f'Passage {i}: ... {p.page_content}' for i, p in enumerate(passages)]) + '...'

        return retrieve_passages

    def page_retrieval_tool(self) -> Tool:
        @tool
        def search_info(query: str, page_name: str) -> str:
            """Use this tool to retrieve information relevant to the given query from a Wikipedia page. Always print and inspect the output of this tool, do not parse its output using regex.
            If you already used this tool, and you didn't get the information you needed, try calling it again with a more specific query and/or a different page.  Don't call this tool twice with the same query/page combination.
            Args:
                query: The query to search for.
                page_name: The name of the Wikipedia page to search in. It must be a valid Wikipedia page name.
            Returns:
                str: The information retrieved.
            """
            self.page = wikipedia.page(page_name, auto_suggest=False)
            self.compute_page_embeddings()
            task = f"""Retrieve information about the query:"{query}" from the Wikipedia page "{self.page.title}"."""
            output = self.passage_search_agent.run(task)
            # reset page
            output = f"Information retrieved from the page '{self.page.title}' for the query '{query}':\n" + output
            self.reset_page_data()
            self.logs.append(format_logs(self.passage_search_agent.get_succinct_logs()))
            return output

        return search_info

    def reset_page_data(self):
        self.page = None
        self.vector_db = None

    def reset(self)->None:
        self.logs = []


class WikipediaRetriever:
    def __init__(self,
                 llm_engine: ModelEngine,
                 page_retriever: PageRetriever,
                 system_prompt: str,
                 search_cfg: SearchCfg = SearchCfg()
                 ):
        self.retrieval_agent = ReactCodeAgent(system_prompt=system_prompt,
                                              llm_engine=llm_engine,
                                              tools=[self.wikipedia_search_tool(),
                                                     page_retriever.page_retrieval_tool()])
        self.search_cfg = search_cfg
        self.logs = []

    def wikipedia_search_tool(self) -> Tool:
        @tool
        def search_wikipedia(query: str) -> str:
            """Use this tool to retrieve pages relevant to the query from Wikipedia. It will return some candidate page names and their summary. Always print and inspect the output of this tool, do not parse its output using regex.
            If you already used this tool, and you didn't get the information you needed, try calling it again with a more specific query. Don't call this tool twice with the same query.
            Args:
                query: The query to search for.
            Returns:
                str: Candidate page names and their summary.
            """
            pages = wikipedia.search(query, results=self.search_cfg.n_pages)
            results = f"Pages found for query '{query}':\n"
            for p in pages:
                try:
                    wiki_page = wikipedia.page(p, auto_suggest=False)
                    results += f"Page: {wiki_page.title}\nSummary: {wiki_page.summary[:self.search_cfg.max_summary_chars]}\n"
                except wikipedia.exceptions.PageError:
                    continue
            return results

        return search_wikipedia

    def reset(self)->None:
        self.logs = []

    def as_tool(self) -> Tool:
        @tool
        def wikipedia_search_agent(query: str) -> str:
            """Use this tool to retrieve information relevant to the given query from Wikipedia. Always print and inspect the output of this tool, do not parse its output using regex.
            If you already used this tool, and you didn't get the information you needed, try calling it again with a more specific query.
            Args:
                query: The query to search for.
            Returns:
                str: The information retrieved.
            """
            output = self.retrieval_agent.run(query)
            self.logs.append(format_logs(self.retrieval_agent.get_succinct_logs()))
            return output

        return wikipedia_search_agent
