# Import the os package
import os

# Import the openai package
import openai
  
OPENAI_API_KEY="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"



def main():
    # Set openai.api_key to the OPENAI environment variable (eg. os.environ["OPENAI"])
    #openai.my_api_key = os.environ["OPENAI"]
    openai.api_key_path = "openai_api_key.txt"

    messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]

    # Define the system message
    system_msg = 'You are a helpful assistant who understands data science.'

    # Define the user message
    user_msg = 'Create a small dataset about total sales over the last year. The format of the dataset should be a data frame with 12 rows and 2 columns. The columns should be called "month" and "total_sales_usd". The "month" column should contain the shortened forms of month names from "Jan" to "Dec". The "total_sales_usd" column should contain random numeric values taken from a normal distribution with mean 100000 and standard deviation 5000. Provide Python code to generate the dataset, then provide the output in the format of a markdown table.'

    # Create a dataset using GPT
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system", "content": system_msg},
                                            {"role": "user", "content": user_msg}])
    
    print(response["choices"][0]["finish_reason"])  #equal to: stop | length | content_filter | null
    print("-------------")
    for choice in response["choices"]:
        print(choice["message"])     
    print("-------------")
    print(response["usage"]["total_tokens"])


if __name__ == "__main__":
    main()