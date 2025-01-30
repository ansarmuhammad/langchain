import os
from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Set your API keys for the respective services
os.environ["TOGETHER_API_KEY"] = "f204369c2f99be55eb5da6efcd3b6b7e9ecd79e55f51a171495e57dbb81bd880"
os.environ["GROQ_API_KEY"] = "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"

# Initialize the ChatTogether LLM for providing improvement suggestions
llm_together = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the ChatGroq LLM for summarization and further improvement
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

# Define the prompt for further improvement
prompt_groq_improvement = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that further improves the given text.",
        ),
        ("human", "{input}"),
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

# Function to further improve the text using Groq
def further_improve_text(improved_text):
    chain_groq_improvement = prompt_groq_improvement | llm_groq
    further_improved_response = chain_groq_improvement.invoke({"input": improved_text})
    return further_improved_response.content

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
improvement_suggestions = get_improvement_suggestions(input_text, summarized_text)

# Step 3: Further improve the text using ChatGroq
further_improved_text = further_improve_text(improvement_suggestions)

# Output the results
print("Summarized Text:", summarized_text)
print("************************ end of summarization ************************")
print("Improvement Suggestions:", improvement_suggestions)
print("************************ end of improvement suggestions ************************")
print("Further Improved Text:", further_improved_text)
