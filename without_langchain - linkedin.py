from groq import Groq
from together import Together
import os

os.environ["TOGETHER_API_KEY"] = "your api key"
os.environ["GROQ_API_KEY"] = "your api key"
# Function to summarize text using Groq model
def summarize_text_with_groq(text):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"You are a helpful assistant that summarizes text. Response should be less than 5 lines.:{text}"}],
        temperature=0,
        max_completion_tokens=None,
        top_p=1,
        stream=True,
        stop=None,
    )
 
    summary = ""
    for chunk in completion:
        summary += (chunk.choices[0].delta.content or "")
 
    return summary

# Function to improve the response using Together model
def improve_response_with_together(original_text, summarized_text):
    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": f"You are a helpful assistant that suggests improvements for the given summarized text.: {summarized_text}. Original text: {original_text}"}
        ]
    )
    
    return response.choices[0].message.content

# Example text to summarize and improve
text_to_summarize = """
New York City, often simply called New York, is the largest city in the United States. It is known for its iconic landmarks such as the Statue of Liberty, Times Square, and Central Park. The city is a major cultural, financial, and media hub, attracting millions of tourists every year. New York City is composed of five boroughs: Manhattan, Brooklyn, Queens, the Bronx, and Staten Island. Visitors can explore the city's diverse neighborhoods, enjoy world-class museums, and experience Broadway shows, making it a popular destination for both locals and tourists alike.
"""

# Step 1: Summarize the text using Groq model
summarized_text = summarize_text_with_groq(text_to_summarize)
print("Summarized Text:", summarized_text)
print("**************************** End of Summarization ***************************")

# Step 2: Improve the summary using Together model
improved_text = improve_response_with_together(text_to_summarize, summarized_text)
print("\nImproved Text:", improved_text)
