import litellm

litellm.set_verbose = True

response = litellm.completion(
    model='openai/gpt4o-opendevin-traj-lora',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
    ],
    base_url='http://cccxc707.pok.ibm.com:8083/v1',
    api_key='fake',
)
print(response)
