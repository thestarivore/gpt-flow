import openai
import datetime
import json
import os


MessageRole = Literal["system", "user", "assistant"]
MessageType = Literal["ai_response", "action_result"]

class MessageDict(dict):
    def __init_(self, role:MessageRole, content):
        self.role = role
        self.content = content

class Message:
    """OpenAI Message object containing a role and the message content"""
    def __init__(self, role:MessageRole, content, message_type:MessageType=None):
        self.role = role
        self.content = content
        self.type = message_type

    def raw(self) -> MessageDict:
        return {"role": self.role, "content": self.content}