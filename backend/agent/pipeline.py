import os
from dotenv import load_dotenv
from typing import Optional, Any

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
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
        self.tools = [retrieve]
        self.model = model or init_chat_model(os.getenv(chat_model_env_var), model_provider="google_genai")
        self.model_with_tools = self.model.bind_tools(self.tools)
        self._build_graph()
        
    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def _agent(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        response = self.model_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
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
        
        self.graph = builder.compile() 

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
        
        for chunk in self.graph.stream({"messages": input_messages}):
            print("Message: ")
            print(chunk)
            if "agent" in chunk and "messages" in chunk["agent"]:
                result = chunk["agent"]["messages"][-1]
        
        print("result: ", result)

        # Logging interaction metrics if available
        if result is not None:        
            return getattr(result, "content", str(result))
        return ""


# Singleton instance for use in FastAPI or elsewhere
rag_pipeline = RAGPipeline()
