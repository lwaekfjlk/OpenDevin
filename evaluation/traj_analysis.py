from collections import defaultdict

import jsonlines


def categorize_merged_data(dataset):
    total_data = defaultdict(int)
    generated_data = defaultdict(int)
    applied_data = defaultdict(int)
    resolved_data = defaultdict(int)
    resolved_instances = defaultdict(list)
    for data in dataset:
        repo = data['instance_id'].split('__')[0]
        resolved = data['fine_grained_report']['resolved']
        applied = data['fine_grained_report']['applied']
        generated = data['fine_grained_report']['generated']
        total_data[repo] += 1
        generated_data[repo] += 1 if generated else 0
        applied_data[repo] += 1 if applied else 0
        resolved_data[repo] += 1 if resolved else 0
        if resolved:
            resolved_instances[repo].append(data['instance_id'])

    data_categorization = {
        'total': total_data,
        'generated': generated_data,
        'applied': applied_data,
        'resolved': resolved_data,
        'resolved_instances': resolved_instances,
    }
    return data_categorization


def categorize_data(dataset):
    total_data = defaultdict(int)
    resolved_data = defaultdict(int)
    resolved_instances = defaultdict(list)
    for data in dataset:
        repo = data['instance_id'].split('__')[0]
        if 'resolved' not in data['test_result']['result']:
            continue
        resolved_data[repo] += data['test_result']['result']['resolved']
        total_data[repo] += 1
        if data['test_result']['result']['resolved'] > 0:
            resolved_instances[repo].append(data['instance_id'])
    data_categorization = {
        'total': total_data,
        'resolved': resolved_data,
        'resolved_instances': resolved_instances,
    }
    # total_num = sum([v for k, v in total_data.items()])
    # resolved_num = sum([v for k, v in resolved_data.items()])
    return data_categorization


if __name__ == '__main__':
    exp_names = [
        'gpt-4o_maxiter_50_N_v1.3',
    ]

    for exp_name in exp_names:
        with jsonlines.open(
            'evaluation_outputs/outputs/swe_bench/CodeActAgent/Qwen2-72B-Instruct_maxiter_50_N_v1.3/output.jsonl',
            'r',
        ) as f:
            qwen2_dataset = [line for line in f]

        qwen2_data_categorization = categorize_data(qwen2_dataset)

        with jsonlines.open(
            f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.merged.jsonl',
            'r',
        ) as f:
            dataset = [line for line in f]

        data_categorization = categorize_merged_data(dataset)

        with jsonlines.open(
            f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output_official.jsonl',
            'r',
        ) as f:
            official_dataset = [line for line in f]

        official_data_categorization = categorize_data(official_dataset)

        for repo in official_data_categorization['total'].keys():
            print('=====================================')
            print(repo)
            print(
                'official release resolved: {}'.format(
                    official_data_categorization['resolved'][repo]
                )
            )
            print(
                'Qwen2 generated release resolved: {}'.format(
                    qwen2_data_categorization['resolved'][repo]
                )
            )
            print(
                'Our generated gpt4o release resolved: {}'.format(
                    data_categorization['resolved'][repo]
                )
            )
            print(
                'total for this repo: {}'.format(
                    official_data_categorization['total'][repo]
                )
            )
            print(
                'official release resolved instances for this repo: {}'.format(
                    official_data_categorization['resolved_instances'][repo]
                )
            )
            print(
                'Qwen generated release resolved instances for this repo: {}'.format(
                    qwen2_data_categorization['resolved_instances'][repo]
                )
            )
            print(
                'Our generated gpt4o resolved instances for this repo: {}'.format(
                    data_categorization['resolved_instances'][repo]
                )
            )
            print(
                'ensemble resolved instances for this repo: {}'.format(
                    len(
                        set(official_data_categorization['resolved_instances'][repo])
                        | set(qwen2_data_categorization['resolved_instances'][repo])
                        | set(data_categorization['resolved_instances'][repo])
                    )
                )
            )
