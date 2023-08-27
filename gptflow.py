# Import the os package
import os

# Import the openai package
import openai

# Import a GPT Assistant
from GPTAssistant import GPTAssistant
from GPTAssistant import OpenAI_GPTAssistant
from web_search import web_search
  
def main():

    # Usage
    assistant = OpenAI_GPTAssistant()
    user_msg = "Find out and write on file the last 5 presidents of UnitedStates"

    response, json_content = assistant.make_first_decision(user_msg)
    assistant.print_response(response)

    if json_content["command"]["name"] == "google":
        print("\n---------------------------------------------\n")
        print(web_search(json_content["command"]["args"]["input"]))


if __name__ == "__main__":
    main()