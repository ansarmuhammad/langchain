import os
from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Set your API keys for the respective services
os.environ["TOGETHER_API_KEY"] = "your together api key"
os.environ["GROQ_API_KEY"] = "your groq api key"

# Initialize the ChatTogehter LLM for providing improvement suggestions

llm_together = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the ChatGroq LLM for summarization
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the prompt for summarization
prompt_groq = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that summarizes text. Response should be less than 5 lines.",
        ),
        ("human", "{input}"),
    ]
)

# Define the prompt for improvement suggestion
prompt_together = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that suggests improvements for the given summarized text.",
        ),
        ("human", "{input_text}\nFirst response: {first_response}"),
    ]
)

# Function to summarize the input text
def summarize_text(input_text):
    chain_groq = prompt_groq | llm_groq
    summary_response = chain_groq.invoke({"input": input_text})
    return summary_response.content

# Function to get suggestions for improvement
def get_improvement_suggestions(input_text, first_response):
    chain_together = prompt_together | llm_together
    improvement_response = chain_together.invoke({
        "input_text": input_text,
        "first_response": first_response
    })
    return improvement_response.content

# Example usage

input_text = ("New York City, often simply called New York, "
              "is the largest city in the United States. "
              "It is known for its iconic landmarks such as the Statue of Liberty, "
              "Times Square, and Central Park. "
              "The city is a major cultural, "
              "financial, and media hub, attracting millions of tourists every year. "
              "New York City is composed of five boroughs: "
              "Manhattan, Brooklyn, Queens, the Bronx, and Staten Island. "
              "Visitors can explore the city's diverse neighborhoods, "
              "enjoy world-class museums, and experience Broadway shows, "
              "making it a popular destination for both locals and tourists alike.")

# Step 1: Summarize the input text using ChatGroq
summarized_text = summarize_text(input_text)


# Step 2: Get improvement suggestions for the summarized text using ChatTogether
improvement_suggestions = get_improvement_suggestions(summarized_text, summarized_text)

# Output the results
print("Summarized Text:", summarized_text)
print("************************ end of summarization ************************")
print("Improvement Suggestions:", improvement_suggestions)
