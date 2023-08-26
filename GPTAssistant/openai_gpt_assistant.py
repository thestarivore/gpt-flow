import openai
import datetime
import json
import os
from GPTAssistant import GPTAssistant

class OpenAI_GPTAssistant(GPTAssistant):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAI_GPTAssistant, cls).__new__(cls)
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
        
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
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
            {"role": "system", "content": f"Only respond in the previoulsy mentioned JSON format."},
            {"role": "user", "content": user_msg},
        ]
        
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        self._increment_token_used(response)
        print(response)
        print(response["choices"][0]["message"]["content"])

        return response["choices"][0]["message"]["content"]
    
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
        
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        self._increment_token_used(response)
        print(response)

        return response["choices"][0]["message"]["content"]

    def print_response(self, response):
        print(response["choices"][0]["finish_reason"])  #equal to: stop | length | content_filter | null
        print("-------------")
        for choice in response["choices"]:
            print(choice["message"])
        print("-------------")
        print(response["usage"]["total_tokens"])
