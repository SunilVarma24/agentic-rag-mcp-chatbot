from typing import List, Dict, Any, Optional
from mcp.mcp_message import MCPMessage


class MCPBus:
    """
    Simple message bus to simulate message passing between agents
    """

    def __init__(self):
        self._queue: List[MCPMessage] = []
        self._history: List[MCPMessage] = []

    def dispatch(
        self,
        sender: str,
        receiver: str,
        msg_type: str,
        payload: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> str:
        
        msg = MCPMessage(sender, receiver, msg_type, payload, trace_id)
        self._queue.append(msg)
        self._history.append(msg)
        print(f"📤 Sent: {msg}")
        return msg.trace_id

    def collect(self, agent: str) -> Optional[MCPMessage]:
        for i, msg in enumerate(self._queue):
            if msg.receiver == agent:
                return self._queue.pop(i)
        return None

    def log(self, trace_id: Optional[str] = None) -> List[MCPMessage]:
        if trace_id:
            return [m for m in self._history if m.trace_id == trace_id]
        return self._history
    
    def get_chat_history(self, trace_id: str) -> List[Dict[str, str]]:
        """
        Returns list of {'query': ..., 'answer': ...} for a given trace_id
        """
        history = []
        for msg in self._history:
            if msg.trace_id == trace_id and msg.msg_type == "final_response":
                question = msg.payload.get("query", "")
                answer = msg.payload.get("response", "")
                if question and answer:
                    history.append({"query": question, "answer": answer})
        return history

    def reset(self):
        self._queue.clear()

# Create a global instance of the message bus
mcp_bus = MCPBus()
