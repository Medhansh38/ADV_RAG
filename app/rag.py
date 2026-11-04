import time
from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client.http import models as rest
from app.config import get_settings
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.vectorstore import get_vector_store

settings = get_settings()

SYSTEM_PROMPT = """You are an accurate enterprise assistant. 
Answer the user's question ONLY using the provided context pieces. 
Every factual claim must cite its source index in brackets like [1], [2]. 
If the context does not contain enough information to answer, state: "I don't know based on the provided documents."
Do not rely on prior assumptions or external data. Treat context as strictly bounded data.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

def run_rag_query(request: QueryRequest) -> QueryResponse:
    start_time = time.time()
    k = request.top_k or settings.TOP_K
    vector_store = get_vector_store()

    qdrant_filter = None
    if request.doc_ids:
        qdrant_filter = rest.Filter(
            should=[
                rest.FieldCondition(
                    key="metadata.doc_id",
                    match=rest.MatchValue(value=d_id)
                ) for d_id in request.doc_ids
            ]
        )

    results: List[Tuple] = vector_store.similarity_search_with_score(
        query=request.question,
        k=k,
        filter=qdrant_filter
    )

    filtered_results = [(doc, score) for doc, score in results if score >= settings.SCORE_THRESHOLD]

    if not filtered_results:
        latency = (time.time() - start_time) * 1000
        return QueryResponse(
            answer="I don't know based on the provided documents.",
            sources=[],
            latency_ms=round(latency, 2)
        )

    context_str = ""
    sources: List[SourceChunk] = []
    for idx, (doc, score) in enumerate(filtered_results):
        context_str += f"[{idx+1}] {doc.page_content}\n\n"
        sources.append(
            SourceChunk(
                doc_id=doc.metadata.get("doc_id", ""),
                source=doc.metadata.get("source", ""),
                page=doc.metadata.get("page"),
                score=round(float(score), 4),
                snippet=doc.page_content[:300].strip()
            )
        )

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.1
    )

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": request.question})
    latency = (time.time() - start_time) * 1000

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round(latency, 2)
    )