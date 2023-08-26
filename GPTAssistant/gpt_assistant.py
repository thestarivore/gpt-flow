import openai
import datetime

class GPTAssistant:
    def initialize(self, name):
        raise NotImplementedError("Subclasses must implement the 'initialize' method.")
    
    def _make_goals(self, user_msg):
        raise NotImplementedError("Subclasses must implement the '_make_goals' method.")
    
    def make_first_decision(self, user_msg):
        raise NotImplementedError("Subclasses must implement the 'make_first_decision' method.")
    
    def make_decision(self, user_msg, relevant_memory, assistant_msg, command_result):
        raise NotImplementedError("Subclasses must implement the 'make_decision' method.")
    
    def print_response(self, response):
        raise NotImplementedError("Subclasses must implement the 'response' method.")

