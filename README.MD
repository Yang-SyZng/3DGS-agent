# 面向 3D 视觉科研的 Agentic RAG 智能分析系统
## 技术栈
LlamaIndex | LangGraph | Agentic RAG | Milvus | Chainlit | RAGAS | Qwen | DeepSeek

Langgraph Workflow

```text
                                         START
                                           ↓
                                      User_Query
                                           ↓
                                        Analyzer
                                    (analyze_query)
                                           ↓
       ┌────── → ────── → ──── → ───── Retrieval
       │                          (retrieve_knowledge)
       │                                   ↓
       │                               Evaluator
       ↑                          (evaluate_retrieval)          
       │                    ┌──────────────┴───────────────┐
       │                    ↓                              ↓
       │                insufficient                   sufficient
       ↑                    ↓                              │
       │              PaperResolver                        │
       │         (resolve_missing_papers)                  │
       │                    ↓                              │
       │           route_zotero_result                     │
       ↑                    ↓                              ↓ 
       │         ┌──────────────────────┐                  │
       │       found                not_found              │  
       │         ↓                      ↓                  │
       │  load_zotero_pdf        search_on_arxiv           │
       ↑         │                 ┌────┴────┐             ↓
       │         │                 ↓         ↓             │
       │         │              matched  unmatched         │
       │         ↓                 │         │             │
       │         │                 ↓     retry/fail        │
       ↑         │          load_arxiv_pdf                 ↓
       │         │                 ↓                       │
       │         └──────────┬──────┘                       │
       │              process_papers                       │
       │                    ↓                              │
       └───── ← ──── update_knowledge                      ↓
                                                           │
                                   ┌───────────────────────┘
                                   ↓
                          ResearchSynthesizer
                          (synthesize_answer)
                                   ↓
                                 Writer
                                   ↓
                                  END
```