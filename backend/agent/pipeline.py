import os
from dotenv import load_dotenv
from typing import Optional, Any
import uuid

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.agent_tools import retrieve

load_dotenv()

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline using LangGraph and LangChain.
    Encapsulates agent graph construction, memory, and tool integration.
    """

    def __init__(
        self,
        model: Any = None,
        tools: Optional[list] = None,
        chat_model_env_var: str = "CHAT_MODEL"
    ):
        """
        Initialize the RAGPipeline.

        Args:
            model: Optional custom chat model. If None, will initialize from environment.
            tools: Optional list of tools. If None, will use default recall tools.
            chat_model_env_var: Environment variable for chat model name.
        """
        print("Initializing Agent...")
        
        with open("backend/agent/prompt.txt", "r", encoding="utf-8") as f:
            self.prompt = f.read()

        self.tools = [retrieve]
        self.model = model or init_chat_model(os.getenv(chat_model_env_var), model_provider="google_genai")
        self.model_with_tools = self.model.bind_tools(self.tools)
        self._build_graph()
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
    def _agent(self, state: MessagesState):
        """Generate answer."""

        # Only process the most recent tool message, if any, otherwise use the last message
        docs_content = None
        tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
        if tool_messages:
            print("Generating response from context...")
            docs_content = "\n\n".join(getattr(msg, "content", "") for msg in reversed(tool_messages))
        else:
            print("Generating response...")

        # Format into prompt

        system_message_content = system_message_content = f"{self.prompt}{docs_content}"
        
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not getattr(message, "tool_calls", False))
        ]
        
        print("Conversation")
        print(conversation_messages
              )
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        
        # Run
        response = self.model_with_tools.invoke(prompt)

        # Ensure tool_calls is not set unless a new tool call is needed
        if hasattr(response, "tool_calls"):
            if not response.tool_calls:
                response.tool_calls = None

        return {"messages": [response]}


    def _route_tools(self, state: MessagesState):
        """
        Tool routing: Determines if tool node should be invoked.
        """
        msg = state["messages"][-1]
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        """
        Build the agent graph with memory, agent, and tool nodes.
        """
        print("Building graph...")
        builder = StateGraph(MessagesState)
        builder.add_node("agent", self._agent)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", self._route_tools, ["tools", END])
        builder.add_edge("tools", "agent")
        memory = MemorySaver()

        self.graph = builder.compile(checkpointer=memory) 

    def ask(
        self,
        query: str
    ) -> str:
        """
        Ask a question to the pipeline and get a response.

        Args:
            query: The user query string.
            user_id: Optional user ID for context.
            thread_id: Optional thread ID for context.

        Returns:
            The agent's response as a string.
        """
        input_messages = [("user", query)]
        result = None
        
        for chunk in self.graph.stream({"messages": input_messages}, config = self.config):
            if "agent" in chunk and "messages" in chunk["agent"]:
                result = chunk["agent"]["messages"][-1]
        
        # Logging interaction metrics if available
        if result is not None:        
            return getattr(result, "content", str(result))
        return ""


# Singleton instance for use in FastAPI or elsewhere
rag_pipeline = RAGPipeline()
