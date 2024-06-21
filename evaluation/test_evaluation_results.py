import jsonlines

exp_names = [
    'claude-2_maxiter_50_N_v1.3', 
    'gpt-4-turbo_maxiter_50_N_v1.3']

for exp_name in exp_names:
    with jsonlines.open(
        f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.jsonl',
        'r',
    ) as f:
        dataset = [line for line in f]

    generated = 0
    applied = 0
    resolved = 0

    valid_dataset = []
    for data in dataset:
        if 'resolved' in data['test_result']['result'].keys():
            valid_dataset.append(data)
        else:
            continue

        if 'test_result' in data.keys() and 'result' in data['test_result']:
            resolved += 1 if data['test_result']['result']['resolved'] > 0 else 0
        if 'git_patch' in data.keys():
            generated += 1 if len(data['git_patch']) > 0 else 0

    with jsonlines.open(
        f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/valid_output.jsonl',
        'w',
    ) as f:
        for data in valid_dataset:
            f.write(data)

    print(exp_name)
    print(f'Generated: {generated}')
    print(f'Resolved: {resolved}')
    print(f'Done: {len(dataset)}')
