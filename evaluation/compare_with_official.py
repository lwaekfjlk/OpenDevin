import json

exp_names = [
    'gpt-4-turbo_maxiter_50_N_v1.3',
    #'Mixtral-8x22B-Instruct-v0.1_maxiter_50_N_v1.3'
    #'Qwen2-72B-Instruct_maxiter_50_N_v1.3',
    #'gemini-1.5-pro-latest_maxiter_50_N_v1.3',
]

MAX_ITER = 50


def process_experiment_files(exp_names, file_suffix, max_iter):
    for exp_name in exp_names:
        with open(
            f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/{file_suffix}',
            'r',
        ) as f:
            data = json.load(f)

        resolved_id = data['resolved']
        return resolved_id


resolved_id = process_experiment_files(exp_names, 'report.json', MAX_ITER)
resolved_id2 = process_experiment_files(exp_names, 'gpt-4-report.json', MAX_ITER)

resolved_id = set(resolved_id)
resolved_id2 = set(resolved_id2)
combined = resolved_id.union(resolved_id2)
print(len(resolved_id), len(resolved_id2), len(combined))
