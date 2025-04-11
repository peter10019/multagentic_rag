from dataclasses import field
from importlib import resources
from multirag import prompts
from pydantic.dataclasses import dataclass


@dataclass
class RetrievalCfg:
    chunk_size: int = 512
    chunk_overlap: int = 256
    n_passages: int = 5
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", ",", " ", ""])
    max_iterations: int = 3


@dataclass
class SearchCfg:
    n_pages: int = 5
    max_summary_chars: int = 1000


@dataclass
class EmbeddingsCfg:
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_model_kwargs: dict = field(default_factory=dict)
    encode_kwargs: dict = field(default_factory=dict)

@dataclass
class EngineCfg:
    summarize_past: bool = True
    max_new_tokens: int = 1024
    temperature: float|None = None
    top_k: int|None = None
    top_p: float|None = None
    do_sample = False

@dataclass
class SystemPrompts:
    manager_system_prompt: str = resources.read_text(prompts, 'manager_system_prompt.txt')
    wikipedia_search_system_prompt: str = resources.read_text(prompts, 'wikipedia_retriever_system_prompt.txt')
    page_retriever_system_prompt: str =  resources.read_text(prompts, 'passage_retriever_system_prompt.txt')