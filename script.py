# Example script for collecting responses from ChatGPT
import openai
import pandas as pd

# Set up OpenAI API
openai.api_key = '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'

def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Load BOLD dataset
bold_data = pd.read_json('profession_prompt.json')  # adjust format as needed

# Collect responses
results = []
for prompt in bold_data['prompts']:
    response = get_chatgpt_response(prompt)
    results.append({'prompt': prompt, 'response': response})

# Save results
pd.DataFrame(results).to_csv('chatgpt_responses.csv', index=False)
