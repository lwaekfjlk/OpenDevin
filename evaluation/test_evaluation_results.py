import jsonlines
from pprint import pprint

exp_names = [
    'gpt-4-turbo_maxiter_50_N_v1.3',
    'Qwen2-72B-Instruct_maxiter_50_N_v1.3',
    'gemini-1.5-pro-latest_maxiter_50_N_v1.3',
]

MAX_ITER = 50

def process_experiment_files(exp_names, file_suffix, max_iter):
    for exp_name in exp_names:
        with jsonlines.open(f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/{file_suffix}', 'r') as f:
            dataset = [line for line in f]

        generated, resolved, total = 0, 0, 0
        jsonline_data = []

        for data in dataset:
            if 'resolved' not in data['test_result']['result']:
                continue
            else:
                total += 1

            if data['history'][-1][0]['message'] == "All done! What's next on the agenda?" or len(data['history']) >= (max_iter + 1):
                jsonline_data.append(data)

            if 'test_result' in data and 'result' in data['test_result']:
                resolved += 1 if data['test_result']['result']['resolved'] > 0 else 0
            if 'git_patch' in data:
                generated += 1 if len(data['git_patch']) > 0 else 0

        with jsonlines.open(f'./{exp_name}_valid.jsonl', 'w') as f:
            for data in jsonline_data:
                f.write(data)

        print(f"{exp_name}")
        print(f"Generated: {generated}")
        print(f"Resolved: {resolved}")
        print(f"Done: {total}")

# Process merged files
def process_merged_experiment_files(exp_names):
    for exp_name in exp_names:
        with jsonlines.open(f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.merged.jsonl', 'r') as f:
            dataset = [line for line in f]

        generated, applied, resolved = 0, 0, 0

        for data in dataset:
            resolved += 1 if data['fine_grained_report']['resolved'] else 0
            generated += 1 if len(data['git_patch']) > 0 else 0
            applied += 1 if data['fine_grained_report']['applied'] else 0
        

        print('===========merged result=============')
        print(f"{exp_name}")
        print(f"Generated: {generated}")
        print(f"Applied: {applied}")
        print(f"Resolved: {resolved}")
        print(f"Done: {len(dataset)}")

if __name__ == '__main__':
    # Process individual and official files
    process_experiment_files(exp_names, 'output.jsonl', MAX_ITER)
    process_experiment_files(exp_names[:1], 'output_official.jsonl', MAX_ITER)
    process_merged_experiment_files(exp_names)
