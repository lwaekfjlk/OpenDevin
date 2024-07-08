import jsonlines


def reformat_history(history):
    new_history = []
    cur_turn = []
    for i, (action, observation) in enumerate(history):
        # Compatibility mode: old format before refractor
        if 'source' not in action:
            return history

        if i == 0:
            assert action['action'] == 'message'
            assert action['source'] == 'user'
            # skip the initial instruction
            continue

        if action['source'] == 'agent':
            # cleanup all previous turns
            if len(cur_turn) == 1:
                new_history.append(cur_turn[0])
            elif len(cur_turn) == 2:
                # one action from user, one action from agent
                agent_msg_action, agent_msg_obs = cur_turn[0]
                assert agent_msg_obs['observation'] == 'null'
                user_msg_action, user_msg_obs = cur_turn[1]
                assert user_msg_obs['observation'] == 'null'
                # re-write user message to be a observation message
                user_msg_action_as_obs = {
                    'observation': 'message',
                    'source': 'user',
                    'content': user_msg_action['args']['content'],
                }
                new_history.append((agent_msg_action, user_msg_action_as_obs))
            elif len(cur_turn) == 0:
                pass
            else:
                raise ValueError(
                    f'Unsupported #interactions per iteration: {len(cur_turn)}'
                )

            # reset new turn
            cur_turn = []
        cur_turn.append((action, observation))
    return new_history


exp_names = [
    #'gpt-3.5-turbo_maxiter_50_N_v1.3',
    #'gpt-4o_maxiter_50_N_v1.3',
    #'gpt-4-turbo_maxiter_50_N_v1.3',
    #'Qwen2-72B-Instruct_maxiter_50_N_v1.3',
    #'gemini-1.5-pro_maxiter_50_N_v1.3',
    #'gemini-1.5-pro-latest_maxiter_50_N_v1.3',
    #'claude-3-opus-20240229_maxiter_50_N_v1.3',
    #'claude-3-5-sonnet-20240620_maxiter_50_N_v1.3',
    'Codestral-22B-v0.1_maxiter_50_N_v1.3'
]

dev_instance_ids = [
    'django__django-10914',
    'django__django-11099',
    'django__django-14382',
    'django__django-14580',
    'django__django-15789',
    'django__django-16527',
    'matplotlib__matplotlib-23964',
    'matplotlib__matplotlib-24334',
    'mwaskom__seaborn-3010',
    'psf__requests-863',
    'pytest-dev__pytest-5227',
    'pytest-dev__pytest-5413',
    'pytest-dev__pytest-7168',
    'sympy__sympy-13480',
    'django__django-13964',
    'django__django-14915',
    'matplotlib__matplotlib-24149',
    'pytest-dev__pytest-11143',
    'scikit-learn__scikit-learn-13142',
    'sphinx-doc__sphinx-8713',
    'sympy__sympy-13647',
    'sympy__sympy-20590',
    'sympy__sympy-23117',
    'sympy__sympy-24213',
]

MAX_ITER = 50


def process_experiment_files(exp_names, file_suffix, max_iter):
    for exp_name in exp_names:
        with jsonlines.open(
            f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/{file_suffix}',
            'r',
        ) as f:
            dataset = [line for line in f]

        # saved_dataset_dict = {}
        # with jsonlines.open(f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output_save.jsonl', 'r') as f:
        #    saved_dataset = [line for line in f]

        # for data in saved_dataset:
        #    saved_dataset_dict[data['instance_id']] = data

        generated, resolved, total, valid = 0, 0, 0, 0
        jsonline_data = []
        oracle_data = []

        for data in dataset:
            if data['instance_id'] not in dev_instance_ids:
                continue
            oracle_datapoint = data.copy()
            oracle_datapoint['git_patch'] = oracle_datapoint['swe_instance']['patch']
            oracle_data.append(oracle_datapoint)
            if 'resolved' not in data['test_result']['result']:
                data['test_result']['result'] = {
                    'test_errored': 0,
                    'test_timeout': 0,
                    'resolved': 0,
                }
            else:
                total += 1

            # if data['history'][-1][0]['message'] == "All done! What's next on the agenda?" or len(data['history']) >= (max_iter + 1):
            #    jsonline_data.append(data)

            formatted_history = reformat_history(data['history'])
            print(len(formatted_history))

            if len(data['git_patch']) > 0:
                jsonline_data.append(data)
                valid += 1

            if 'test_result' in data and 'result' in data['test_result']:
                resolved += 1 if data['test_result']['result']['resolved'] > 0 else 0
                if data['test_result']['result']['resolved'] > 0:
                    print(data['instance_id'])
            if 'git_patch' in data:
                generated += 1 if len(data['git_patch']) > 0 else 0

        with jsonlines.open(f'./{exp_name}_valid.jsonl', 'w') as f:
            for data in jsonline_data:
                f.write(data)

        with jsonlines.open(f'./{exp_name}_oracle.jsonl', 'w') as f:
            for data in oracle_data:
                f.write(data)

        print(f'{exp_name}')
        print(f'Generated: {generated}')
        print(f'Resolved: {resolved}')
        print(f'Valid: {valid}')
        print(f'Done: {total}')


# Process merged files
def process_merged_experiment_files(exp_names):
    for exp_name in exp_names:
        with jsonlines.open(
            f'evaluation_outputs/outputs/swe_bench/CodeActAgent/{exp_name}/output.merged.jsonl',
            'r',
        ) as f:
            dataset = [line for line in f]

        generated, applied, resolved = 0, 0, 0

        for data in dataset:
            resolved += 1 if data['fine_grained_report']['resolved'] else 0
            generated += 1 if len(data['git_patch']) > 0 else 0
            applied += 1 if data['fine_grained_report']['applied'] else 0

        print('===========merged result=============')
        print(f'{exp_name}')
        print(f'Generated: {generated}')
        print(f'Applied: {applied}')
        print(f'Resolved: {resolved}')
        print(f'Done: {len(dataset)}')


if __name__ == '__main__':
    # Process individual and official files
    process_experiment_files(exp_names, 'output.jsonl', MAX_ITER)
    # process_experiment_files(exp_names[:1], 'output_official.jsonl', MAX_ITER)
    # process_merged_experiment_files(exp_names)
