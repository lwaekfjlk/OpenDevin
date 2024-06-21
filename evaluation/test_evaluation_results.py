import jsonlines
from pprint import pprint

exp_names = [
    #'claude-3-opus-20240229_maxiter_50_N_v1.3',
    #'gemini-1.5-pro-latest_maxiter_50_N_v1.3',
    #'gpt-4o_maxiter_50_N_v1.3',
    'gpt-4-turbo_maxiter_50_N_v1.3',
    #'gpt-3.5-turbo_maxiter_50_N_v1.3',
    #'Mixtral-8x22B-Instruct-v0.1_maxiter_50_N_v1.3',
    'Qwen2-72B-Instruct_maxiter_50_N_v1.3',
    #'claude-3-5-sonnet-20240620_maxiter_50_N_v1.3'
]

MAX_ITER = 50

for exp_name in exp_names:
    with jsonlines.open(
        f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.jsonl',
        'r',
    ) as f:
        dataset = [line for line in f]

    generated = 0
    applied = 0
    resolved = 0

    jsonline_data = []
    for data in dataset:
        if 'resolved' not in data['test_result']['result'].keys():
            continue

        if data['history'][-1][0]['message'] == 'All done! What\'s next on the agenda?' or len(data['history']) >= (MAX_ITER + 1):
            jsonline_data.append(data)
        #else:
        #    continue
        if 'test_result' in data and 'result' in data['test_result']:
            resolved += 1 if data['test_result']['result']['resolved'] > 0 else 0
        if 'git_patch' in data.keys():
            generated += 1 if len(data['git_patch']) > 0 else 0

    with jsonlines.open(f'./{exp_name}_valid.jsonl', 'w') as f:
        for data in jsonline_data:
            f.write(data)

    print(exp_name)
    print(f'Generated: {generated}')
    print(f'Resolved: {resolved}')
    print(f'Done: {len(dataset)}')

exp_names = [
    'gpt-4-turbo_maxiter_50_N_v1.3',
    'Qwen2-72B-Instruct_maxiter_50_N_v1.3',
]


for exp_name in exp_names:
    with jsonlines.open(
        f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.merged.jsonl',
        'r',
    ) as f:
        dataset = [line for line in f]

    generated = 0
    applied = 0
    resolved = 0
    applied = 0

    for i, data in enumerate(dataset):
        resolved += 1 if data['fine_grained_report']['resolved'] else 0
        generated += 1 if len(data['git_patch']) > 0 else 0
        applied += 1 if data['fine_grained_report']['applied'] else 0

    print('===========merged result=============')
    print(exp_name)
    print(f'Generated: {generated}')
    print(f'Applied: {applied}')
    print(f'Resolved: {resolved}')
    print(f'Done: {len(dataset)}')
