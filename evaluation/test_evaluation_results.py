import jsonlines

exp_names = ['claude-2_maxiter_50_N_v1.3', 'gemini-1.5-flash_maxiter_50_N_v1.3']

for exp_name in exp_names:
    with jsonlines.open(
        f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.jsonl',
        'r',
    ) as f:
        dataset = [line for line in f]

    generated = 0
    applied = 0
    resolved = 0

    for data in dataset:
        if 'test_result' in data and 'result' in data['test_result']:
            resolved += 1 if data['test_result']['result']['resolved'] > 0 else 0
        if 'git_patch' in data:
            generated += 1 if len(data['git_patch']) > 0 else 0

    print(exp_name)
    print(f'Generated: {generated}')
    print(f'Resolved: {resolved}')
    print(f'Done: {len(dataset)}')
