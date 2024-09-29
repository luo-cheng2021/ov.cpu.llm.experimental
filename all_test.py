import argparse
import glob
import shutil
import subprocess
import sys
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
    'int8',
    'f32'
]

def get_test_info(prec):
    infos = {
        # prec: (dir, infer_prec_hint)
        'bf16': ('f16', 'bf16'),
        'f16' : ('f16', 'f16'),
        'int8': ('SQ', 'bf16'),
        'f32' : ('f16', 'f32') 
    }
    return infos[prec]

f = open('all_test.log', 'w')

def test_perf():
    os.environ['LLM_DQ'] = '3'
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
                if result.returncode:
                    print(f'return code: {result.returncode} failed: {result.stderr.decode("utf-8")}')
                    raise Exception(f'test {cmd} failed')
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

def test_accuary():
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
            cmd = f'numactl -C0-31 python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path={model}{info[0]},nbatch=1,prec={info[1]}' # -L 100'
            print(f'test {prec:<4} "{cmd}"...', end='', flush=True)
            beg = time.time()
            result = subprocess.run(cmd.split(), capture_output=True)
            end = time.time()
            if result.returncode:
                print(f'return code: {result.returncode} failed: {result.stderr.decode("utf-8")}')
                raise Exception(f'test {cmd} failed')
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

def convert():
    cmds = [
        'python -m ovllm.export.llama --org_model_path meta-llama/Llama-2-13b-hf --ov_model_path ./gen/llama-2-13b/',
        'python -m ovllm.export.llama --org_model_path meta-llama/Llama-2-7b-hf --ov_model_path ./gen/llama-2-7b/',
        'python -m ovllm.export.chatglm3 --org_model_path THUDM/chatglm3-6b --ov_model_path ./gen/chatglm3-6b/',
        'python -m ovllm.export.gptj --org_model_path EleutherAI/gpt-j-6b --ov_model_path ./gen/gptj_6b/'
    ]
    # order should be same with above
    quant_configs = [
        'sq_config_llama2_13b.yaml',
        'sq_config_llama2_7b.yaml',
        'sq_config_chatglm3_6b.yaml',
        'sq_config_gptj_6b.yaml'
    ]
    all_beg = time.time()
    def run_cmd(cmd):
        beg = time.time()
        result = subprocess.run(cmd.split(), capture_output=True)
        end = time.time()
        out = result.stdout.decode("utf-8")
        if result.returncode:
            print(f'return code: {result.returncode} failed: {result.stderr.decode("utf-8")}')
            raise Exception(f'convert {cmd} failed')
        else:
            print(f'cost {end - beg:.2f} seconds')

        f.write(out)

    for idx, cmd in enumerate(cmds):
        # f16 model
        full_cmd = f'{cmd} --quant f16'
        print(f'convert f16: "{full_cmd}"... ', end='', flush=True)
        run_cmd(full_cmd)

        # int8 model
        model_dir = cmd.split('--ov_model_path')[1].strip()
        f16_dir = f"{model_dir}f16/"
        print(f'convert int8 model... ')
        cmd_calib = f'python -m ovllm.sq_calibration  -m {f16_dir} model.pickle'
        print(f'calibration: "{cmd_calib}"... ', end='', flush=True)
        run_cmd(cmd_calib)
        cmd_quant = f'python -m ovllm.sq_quant -m={f16_dir}openvino_model.xml -s model.pickle -c={quant_configs[idx]} {model_dir}SQ/openvino_model.xml'
        print(f'quant: "{cmd_quant}"... ', end='', flush=True)
        run_cmd(cmd_quant)
        cmd_cp = f'cp {f16_dir}*.json {f16_dir}*.model {model_dir}SQ/'
        print(f'cp: "{cmd_cp}"... ', flush=True)
        [shutil.copy(src, f'{model_dir}SQ/') for src in glob.glob(f'{f16_dir}*.json')]
        [shutil.copy(src, f'{model_dir}SQ/') for src in glob.glob(f'{f16_dir}*.model')]

    all_end = time.time()
    print(f'all cost {all_end - all_beg:.1f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument("-c", "--convert", action="store_true")
    parser.add_argument("-p", "--test-performance", action="store_true")
    parser.add_argument("-a", "--test-accuary", action="store_true")
    # Parse the argument
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    if args.convert:
        convert()

    if args.test_performance:
        test_perf()

    if args.test_accuary:
        test_accuary()
