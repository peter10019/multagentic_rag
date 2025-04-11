from langchain_huggingface import HuggingFaceEmbeddings
from transformers import PreTrainedModel, PreTrainedTokenizerFast, ReactCodeAgent

from multirag.agents import WikipediaRetriever, PageRetriever
from multirag.cfg import EngineCfg, EmbeddingsCfg, SystemPrompts, RetrievalCfg, SearchCfg
from multirag.engine import ModelEngine
from multirag.utils import format_logs


class MultiAgenticRAG:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 prompts: SystemPrompts = SystemPrompts(),
                 engine_cfg: EngineCfg = EngineCfg(),
                 embeddings_cfg: EmbeddingsCfg = EmbeddingsCfg(),
                 retrieval_cfg: RetrievalCfg = RetrievalCfg(),
                 search_cfg: SearchCfg = SearchCfg()):
        self.llm_engine = ModelEngine(model=model,
                                      tokenizer=tokenizer,
                                      summarize_past=engine_cfg.summarize_past,
                                      max_new_tokens=engine_cfg.max_new_tokens,
                                      temperature=engine_cfg.temperature,
                                      top_k=engine_cfg.top_k,
                                      top_p=engine_cfg.top_p,
                                      do_sample=engine_cfg.do_sample)
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_cfg.embedding_model_name,
                                                **embeddings_cfg.embedding_model_kwargs)
        self.page_retriever = PageRetriever(llm_engine=self.llm_engine,
                                            embeddings=self.embeddings,
                                            system_prompt=prompts.page_retriever_system_prompt,
                                            retrieval_cfg=retrieval_cfg)
        self.wikipedia_retriever = WikipediaRetriever(llm_engine=self.llm_engine,
                                                      page_retriever=self.page_retriever,
                                                      system_prompt=prompts.wikipedia_search_system_prompt,
                                                      search_cfg=search_cfg)
        self.manager_agent = ReactCodeAgent(system_prompt=prompts.manager_system_prompt,
                                            llm_engine=self.llm_engine,
                                            tools=[self.wikipedia_retriever.as_tool()])

    @property
    def manager_agent_logs(self) -> dict[str, any]:
        return format_logs(self.manager_agent.get_succinct_logs())

    @property
    def wikipedia_search_agent_logs(self) -> list[dict[str, any]]:
        return self.wikipedia_retriever.logs

    @property
    def page_search_agent_logs(self) -> list[dict[str, any]]:
        return self.page_retriever.logs

    def __call__(self, task: str) -> str:
        self.wikipedia_retriever.reset()
        self.page_retriever.reset()
        return self.manager_agent.run(task=task)
