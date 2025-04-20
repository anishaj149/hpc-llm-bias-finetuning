# Example script for collecting responses from ChatGPT
import openai
import pandas as pd

# Set up OpenAI API
openai.api_key = 'your-api-key'

def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Load BOLD dataset
bold_data = pd.read_csv('bold.csv')  # adjust format as needed

# Collect responses
results = []
for prompt in bold_data['prompts']:
    response = get_chatgpt_response(prompt)
    results.append({'prompt': prompt, 'response': response})

# Save results
pd.DataFrame(results).to_csv('chatgpt_responses.csv', index=False)