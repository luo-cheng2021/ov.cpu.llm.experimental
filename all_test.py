import subprocess
import time
import os

models = {
    './gen/llama-2-7b/': [
        # batch, in, out
        {'b': 1,  'in': 1024, 'out': 128},
        {'b': 1,  'in': 2016, 'out': 32},
        {'b': 32, 'in': 1024, 'out': 128},
        {'b': 16, 'in': 2016, 'out': 32},
    ],
    './gen/llama-2-13b/': [
        # batch, in, out
        {'b': 1, 'in': 1024, 'out': 128},
        {'b': 1, 'in': 2016, 'out': 32},
        {'b': 30,'in': 1024, 'out': 128, 'b_f32': 16},
        {'b': 16, 'in': 2016, 'out': 32, 'b_f32': 8},
    ],
    './gen/gptj_6b/': [
        # batch, in, out
        {'b': 1, 'in': 1024, 'out': 128},
        {'b': 1, 'in': 2016, 'out': 32},
        {'b': 32,'in': 1024, 'out': 128},
        {'b': 16,'in': 2016, 'out': 32},
    ],
    './gen/chatglm3-6b/': [
        # batch, in, out
        {'b': 1, 'in': 1024, 'out': 128},
        {'b': 1, 'in': 2016, 'out': 32},
        {'b': 32,'in': 1024, 'out': 128},
        {'b': 16,'in': 2016, 'out': 32},
    ],
}

precs = [
    'bf16',
    'f16',
    #'int8',
    'f32'
]

def get_test_info(prec):
    infos = {
        # prec: (dir, infer_prec_hint)
        'bf16': ('f16', 'bf16'),
        'f16' : ('f16', 'f16'),
        'int8': ('INT8_SYM', 'bf16'),
        'f32' : ('f16', 'f32') 
    }
    return infos[prec]

f = open('all_test.log', 'w')

def test_perf():
    results = {}
    all_beg = time.time()
    for (model, token_infos) in models.items():
        if os.path.exists(model):
            print(f'testing {model}...')
        else:
            print(f'{model} not exist, skipped')
            continue
        results[model] = {}
        for token_info in token_infos:
            print(f'testing {token_info}...')
            batch = token_info['b']
            in_len = token_info['in']
            out_len = token_info['out']
            for prec in precs:
                new_batch = batch
                if f'b_{prec}' in token_info:
                    new_batch = token_info[f'b_{prec}']
                info = get_test_info(prec)
                cmd = f'python -m ovllm -m {model}{info[0]} -pl {new_batch}x{in_len} -al {out_len}  -r 2 -bs 4 --prec {info[1]} --numa 0 1 2 3 4 5'
                print(f'test {prec:<4} "{cmd}"...', end='', flush=True)
                beg = time.time()
                result = subprocess.run(cmd.split(), capture_output=True)
                end = time.time()
                out = result.stdout.decode("utf-8")
                cur_result = []
                for line in out.split('\n'):
                    if 'Total FPS' in line:
                        cur_result.append(line.split('thoughput): ')[1])
                token_len = (new_batch, in_len, out_len)
                if token_len not in results[model]:
                    results[model][token_len] = {}
                results[model][token_len][prec] = cur_result
                print(f'fps: {", ".join(cur_result)}, cost: {end - beg:.1f} seconds')
                #print(out)
                f.write(out)

    all_end = time.time()
    import pprint
    print(f'all cost {all_end - all_beg:.1f} seconds:')
    def pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    pretty(results)

def test_accuracy():
    results = {}
    all_beg = time.time()
    for model in models.keys():
        if os.path.exists(model):
            print(f'testing {model}...')
        else:
            print(f'{model} not exist, skipped')
            continue
        results[model] = {}
        for prec in precs:
            info = get_test_info(prec)
            cmd = f'python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path={model}{info[0]},nbatch=1,prec={info[1]}' # -L 100'
            print(f'test {prec:<4} "{cmd}"...', end='', flush=True)
            beg = time.time()
            result = subprocess.run(cmd.split(), capture_output=True)
            end = time.time()
            out = result.stdout.decode("utf-8")
            cur_result = []
            for line in out.split('\n'):
                if 'lambada_openai' in line:
                    cur_result.append(float(line.split('|acc')[1].split('|')[2]) * 100)
                if 'perplexity|' in line:
                    cur_result.append(float(line.split('perplexity|')[1].split('|')[1]))
            results[model][prec] = cur_result
            print(f'acc,ppl: {cur_result}, cost: {end - beg:.1f} seconds')
            #print(out)
            f.write(out)

    all_end = time.time()
    import pprint
    print(f'all cost {all_end - all_beg:.1f} seconds:')
    def pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    pretty(results)

#test_perf()
test_accuracy()

#pprint.pprint(results)