import ollama

client=ollama.Client()

model='auscounsellor'
prompt='Top International Diaspora at Australia'

response=client.generate(model=model,prompt=prompt)

print("Response from ollama")
print(response.response)
