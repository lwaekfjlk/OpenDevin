import litellm

litellm.set_verbose = True

headers = {
    'Content-Type': 'application/json',
}

response = litellm.completion(
    model='openai/./Qwen2-7B-Instruct',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
    ],
    base_url='http://cccxc710.pok.ibm.com:8000/v1/',
    headers=headers,
)

print(response)