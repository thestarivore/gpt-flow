import openai
import datetime
import json
import os
from llama_index import GPTTreeIndex

import gpt_agent

from typing import TYPE_CHECKING, List, Literal, Optional, TypedDict

from utils import count_message_tokens, count_string_tokens

GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"

class OpenAI_GPT_Agent(GPT_Agent):
    _instance = None
    GPT_GOALS_MAKING_TEMPERATURE = 1.0
    GPT_DECISION_MAKING_TEMPERATURE = 1.0
    GPT_SUMMARY_MAKING_TEMPERATURE = 1.0

    # Define the expected structure for the thoughts object of the response JSON
    _expected_thoughts_structure = {
        "text": str,
        "reasoning": str,
        "plan": str,
        "criticism": str,
        "speak": str
    }
    # Define the possible argument value of the command described in the JSON response
    _possible_command_args_values = ["input", "url", "question", "file", "text", "directory", "reason"]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAI_GPT_Agent, cls).__new__(cls)
            cls._instance.initialize("GPT-FLOW")
        return cls._instance
    
    def initialize(self, name):
        # Set openai.api_key to the OPENAI environment variable (eg. os.environ["OPENAI"])
        #openai.my_api_key = os.environ["OPENAI"]
        script_dir = os.path.dirname(__file__)
        openai.api_key_path = os.path.join(script_dir, "openai_api_key.txt")
        self.name = name
        self.token_used = 0
        
        # Load system messages from file
        with open(os.path.join(script_dir, "base_system_message.txt"), "r") as file:
            self.base_system_msg = file.read().strip()
        with open(os.path.join(script_dir, "goals_system_message.txt"), "r") as file:
            self.goals_system_msg = file.read().strip()
        with open(os.path.join(script_dir, "goals_user_message.txt"), "r") as file:
            self.goals_user_msg = file.read().strip()

    def _increment_token_used(self, response):
        self.token_used = self.token_used + response["usage"]["total_tokens"]

    def _make_goals(self, user_msg):     
        '''
        This will produce a JSON containing up to 5 goals related to the user's request.
        The sytax will be the following:
        {
            "goals": [
                "goal 1 short decision",
                "goal 2 short decision",
                "etc.",
            ]
        } 
        ''' 
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Compose system message
        data = {'Name': self.name}
        system_msg = self.goals_system_msg.format(**data)
        data = {'UserMessage': user_msg}
        complete_user_msg = self.goals_user_msg.format(**data)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "system", "content": f"Current date and time: {current_datetime}"},
            {"role": "user", "content": complete_user_msg}
        ]
        
        response = openai.ChatCompletion.create(model=GPT_3_MODEL, messages=messages, temperature=self.GPT_GOALS_MAKING_TEMPERATURE)
        self._increment_token_used(response)
        print(response)

        # Compose goals string
        i = 1
        goals = ""
        rsp_content = json.loads(response["choices"][0]["message"]["content"])
        for goal in rsp_content["goals"]:
            goals = goals + str(i) + ". " + goal + "\n"
            i = i + 1

        return goals

    def _is_valid_json(self, json_string):
        '''
        Check if the string passed as argument is a valid JSON
        '''
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    def _has_expected_structure(self, json_obj):
        '''
        Check if the JSON object passed as argument has the expected structure and arguments
        '''
        if "thoughts" in json_obj and "command" in json_obj:
            # Check "thoughts" object
            thoughts_obj = json_obj["thoughts"]
            for key, value_type in self._expected_thoughts_structure.items():
                if key not in thoughts_obj or not isinstance(thoughts_obj[key], value_type):
                    return False
            
            # Check "command" object
            command_obj = json_obj["command"]
            if "name" in command_obj and "args" in command_obj:
                command_args_obj = command_obj["args"]
                for key in command_args_obj:
                    if not key in self._possible_command_args_values:
                        return False
            else:
                return False
        else:
            return False
        
        return True
    
    def _make_completion_with_vaild_response(self, messages, temperature):
        '''
        Make a OpenAI completion and repeat it until we get a valid JSON in response
        '''
        response, content, json_content = None, None, None
        while True:
            response = openai.ChatCompletion.create(model=GPT_3_MODEL, messages=messages, temperature=temperature)
            self._increment_token_used(response)
            print(response)
            content = response["choices"][0]["message"]["content"]
            #print(content)
            
            if(self._is_valid_json(content)):
                json_content = json.loads(content)
                if (self._has_expected_structure(json_content)):
                    print ("Got a VALID JSON response!\n")
                    print(content)
                    break
                else:
                    print ("Got an INVALID JSON response!\n")
            else:
                print ("Got an INVALID JSON response!\n")

        return response, json_content
    
    def make_first_decision(self, user_msg): 
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate Goals based on user's request
        self.goals = self._make_goals(user_msg)

        # Compose system message
        data = {'Name': self.name, 'Goals': self.goals}
        self.system_msg = self.base_system_msg.format(**data)

        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "system", "content": f"Current date and time: {current_datetime}"},
            {"role": "system", "content": f"This reminds you of these events from your past: "},
            {"role": "system", "content": f"Only respond in the previoulsy mentioned JSON format, containing 'thoughts' and 'command'."},
            {"role": "user", "content": user_msg},
        ]
        
        return self._make_completion_with_vaild_response(messages, self.GPT_DECISION_MAKING_TEMPERATURE)
    
    def make_decision(self, relevant_memory, last_user_msg, assistant_msg, command_result, user_msg):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "system", "content": f"Current date and time: {current_datetime}"},
            {"role": "system", "content": f"This reminds you of these events from your past: {relevant_memory}"},
            {"role": "user", "content": last_user_msg},
            {"role": "assistant", "content": assistant_msg},
            {"role": "system", "content": f"Last command result: {command_result}"},
            {"role": "user", "content": user_msg}
        ]
        
        return self._make_completion_with_vaild_response(messages, self.GPT_DECISION_MAKING_TEMPERATURE)

    def make_summary(self, user_msg: str, text_to_summarize: str, base_summary: str = None) -> tuple[str, str]:
        SINGLE_SUMMARY_CAPACITY: int = 7000
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        messages = [
            {"role": "system", "content": """The user is trying to find more information about: {user_msg}\n
                                            Focus on information related the user's subject, omit personal information 
                                            contained in the text""".format(user_msg=user_msg)},
            {"role": "system", "content": f"Current date and time: {current_datetime}"},
            #{"role": "user", "content": "Make a summary of the following text: \n"+text_to_summarize+""},
        ]

        #Count current messages Tokens
        tokens_until_now: int = count_message_tokens(messages=messages, model="gpt-3.5-turbo")
        tokens_text_to_summarize: int = count_string_tokens(string=text_to_summarize, model_name="gpt-3.5-turbo")

        query = """The user is trying to find more information about: {user_msg}\n
                   Summarize the text focusing on information related the user's subject and omit personal information 
                   contained in the text""".format(user_msg=user_msg)
        index = GPTTreeIndex(text_to_summarize)
        response = index.query(query, mode="summarize")



        #------------------------------------------------------------

        # Check number of tokens
        text_length: int = len(text_to_summarize)       #TODO: count nr of tokens instead of characters
        base_text_length: int = 0
        if base_summary is not None:
            base_text_length = len(base_summary)
        remaining_text_to_summarize: str = ""
        text: str = ""
        must_split_summary: bool = False
        if (text_length + base_text_length) > SINGLE_SUMMARY_CAPACITY:
            text = text_to_summarize[:(SINGLE_SUMMARY_CAPACITY - base_text_length)]
            remaining_text_to_summarize = text_to_summarize[(SINGLE_SUMMARY_CAPACITY - base_text_length):]
            must_split_summary = True
        else:
            text = text_to_summarize

        # If we don't have the previous summary has been provided
        if base_summary is not None:
            text = base_summary + "\n" + text
        messages.append({"role": "user", "content": "Make a summary of the following text: \n" + text + ""})

        response = openai.ChatCompletion.create(model=GPT_3_MODEL,
                                                messages=messages,
                                                temperature=self.GPT_SUMMARY_MAKING_TEMPERATURE)
        self._increment_token_used(response)
        print(response)
        content = response["choices"][0]["message"]["content"]
        print(content)

        # Recursive call if we must split the summary
        if must_split_summary:
            return self.make_summary(user_msg=user_msg,
                                     text_to_summarize=remaining_text_to_summarize,
                                     base_summary=content)
        else:
            return response, content

    def print_response(self, response):
        print(response["choices"][0]["finish_reason"])  #equal to: stop | length | content_filter | null
        print("-------------")
        for choice in response["choices"]:
            print(choice["message"])
        print("-------------")
        print(response["usage"]["total_tokens"])
