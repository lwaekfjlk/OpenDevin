import litellm

litellm.set_verbose = True


response = litellm.completion(
    model='openai/Mixtral-8x22B-Instruct-v0.1',
    messages=[
        # {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
    ],
    base_url='http://cccxc710.pok.ibm.com:8081/v1',
    api_key='fake',
)
print(response)
