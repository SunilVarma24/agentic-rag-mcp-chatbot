from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime


class MCPMessage:
    """
    Model Context Protocol Message Structure
    """

    def __init__(
        self,
        sender: str,
        receiver: str,
        msg_type: str,
        payload: Dict[str, Any],
        trace_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.sender = sender
        self.receiver = receiver
        self.msg_type = msg_type
        self.trace_id = trace_id or str(uuid4())[:8]
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.msg_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    def __repr__(self):
        return f"[{self.msg_type}] {self.sender} → {self.receiver} | trace_id={self.trace_id}"
