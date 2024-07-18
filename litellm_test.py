import litellm

litellm.set_verbose = True


response = litellm.completion(
    model='openai/commit-pack-lora',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
    ],
    base_url='http://cccxc709.pok.ibm.com:8084/v1',
    api_key='fake',
)
print(response)
