
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
import json
import uuid
from datetime import datetime

from utils.embeddings import ChromaVectorStore
from utils.openai_client import OpenAIClient
from database.models import TelemetryLog, SessionLocal
from config.settings import Config

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    retrieved_documents: List[dict]
    context: str
    response: str
    session_id: str
    transaction_id: str
    route_decision: str

class AgenticRAGBot:
    def __init__(self):
        self.vector_store = ChromaVectorStore()
        self.openai_client = OpenAIClient()
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the LangGraph workflow"""
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("generation", self._generation_node)
        workflow.add_node("direct_response", self._direct_response_node)

        # Add edges
        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "retrieval": "retrieval",
                "direct": "direct_response"
            }
        )
        workflow.add_edge("retrieval", "generation")
        workflow.add_edge("generation", END)
        workflow.add_edge("direct_response", END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        """Router node to decide the flow"""
        query = state["query"]

        # Create routing prompt
        routing_prompt = f"""
        You are a routing agent. Analyze the user query and decide whether it needs:
        1. "retrieval" - if the query requires specific information from a knowledge base
        2. "direct" - if it's a greeting, general question, or can be answered directly

        Query: {query}

        Respond with only "retrieval" or "direct".
        """

        messages = [SystemMessage(content=routing_prompt)]
        decision = self.openai_client.generate_structured_response(messages)

        # Default to direct response if decision is not valid
        route_decision = "direct"
        if isinstance(decision, str) and "retrieval" in decision.lower():
            route_decision = "retrieval"

        state["route_decision"] = route_decision
        return state

    def _retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieval node to get relevant documents"""
        query = state["query"]

        # Retrieve documents from vector store
        documents = self.vector_store.similarity_search(query, k=5)

        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}: {doc['content']}")

        context = "\n\n".join(context_parts)

        state["retrieved_documents"] = documents
        state["context"] = context
        return state

    def _generation_node(self, state: AgentState) -> AgentState:
        """Generation node for RAG-based responses"""
        query = state["query"]
        context = state["context"]

        # Create generation prompt
        system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so politely.

        Context:
        {context}
        """

        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=query)
        ]

        response = self.openai_client.generate_response(messages)
        state["response"] = response

        # Log telemetry
        self._log_telemetry(state)

        return state

    def _direct_response_node(self, state: AgentState) -> AgentState:
        """Direct response node for simple queries"""
        query = state["query"]

        # Create direct response prompt
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Respond naturally to the user's message."),
            HumanMessage(content=query)
        ]

        response = self.openai_client.generate_response(messages)
        state["response"] = response

        # Log telemetry (with empty documents)
        state["retrieved_documents"] = []
        state["context"] = ""
        self._log_telemetry(state)

        return state

    def _route_decision(self, state: AgentState) -> str:
        """Conditional edge function"""
        return state["route_decision"]

    def _log_telemetry(self, state: AgentState):
        """Log interaction to telemetry database"""
        try:
            db = SessionLocal()

            # Format retrieved documents for storage
            retrieved_docs = []
            for doc in state.get("retrieved_documents", []):
                retrieved_docs.append({
                    "content": doc.get("page_content", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                })

            # Create complete prompt (approximation)
            complete_prompt = f"Query: {state['query']}\nContext: {state.get('context', '')}"

            telemetry_log = TelemetryLog(
                transaction_id=state["transaction_id"],
                session_id=state["session_id"],
                query=state["query"],
                retrieved_documents=retrieved_docs,
                complete_prompt=complete_prompt,
                response=state["response"]
            )

            db.add(telemetry_log)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Error logging telemetry: {e}")

    def process_query(self, query: str, session_id: str = None) -> dict:
        """Process a user query through the agent"""
        # Generate IDs
        transaction_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create initial state
        initial_state = {
            "messages": [],
            "query": query,
            "retrieved_documents": [],
            "context": "",
            "response": "",
            "session_id": session_id,
            "transaction_id": transaction_id,
            "route_decision": ""
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        return {
            "response": result["response"],
            "transaction_id": transaction_id,
            "session_id": session_id,
            "retrieved_documents": result["retrieved_documents"],
            "route_taken": result["route_decision"]
        }
