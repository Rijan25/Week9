import requests
import json

# Setting up the base URL for the local Ollama API
url='http://localhost:11434/api/chat'

# Define the payload (input prompt)
payload={
    'model':'llama2',
    'messages':[{'role':'user','content':'How can I crack Remote AI Jobs?'}]
}

response=requests.post(url,json=payload,stream=True)

if response.status_code==200:
    print('Fetching the requests from the Ollama')
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data=json.loads(line)
                if 'message' in json_data and 'content' in json_data['message']:
                    print(json_data['message']['content'],end='')
            except json.JSONDecodeError:
                print(f'Failed to parse the line:{line}')
    print()
else:
    print(f'Error:{response.status_code}')    



