import os
import time
import pickle
import argparse
import base64, zlib
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import random
import sys

import tflib as lib
import models
import utils
from pcfg_cracker.lib_guesser.pcfg_grammar import PcfgGrammar

def set_GPU():
    """GPU相关设置"""

    # 打印变量在那个设备上
    # tf.debugging.set_log_device_placement(True)
    # 获取物理GPU个数
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('物理GPU个数为：', len(gpus))
    # 设置内存自增长
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('-------------已设置完GPU内存自增长--------------')
    # 获取逻辑GPU个数
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus))

def pcfg_grammer():
    
    program_info = {
        # Program and Contact Info
        'name':'PRINCE-LING',
        'version': '4.3',
        'author':'Matt Weir',
        'contact':'cweir@vt.edu',

        # Standard Options
        'rule_name':'result',
        'output_file':'gen.txt',
        'max_size':None,

        # Advanced Options
        'skip_case':False,
        }
    base_directory = os.path.join(os.path.realpath("."),'pcfg_cracker/Rules',program_info['rule_name'])
    pcfg = PcfgGrammar(
                program_info['rule_name'],
                base_directory,
                program_info['version'],
                base_structure_folder = "Prince",
                skip_case = program_info['skip_case'],
                )
    
    return pcfg
def generate_samples(session, fake_inputs, charmap, inv_charmap):
    samples = session.run(fake_inputs)
    #samples = np.argmax(samples, axis=2)
    
    selected_classes = np.zeros((samples.shape[0], samples.shape[1]), dtype=int)
    # 遍历 batch 和 sequence length 进行选择
    for i in range(samples.shape[0]):  # 遍历每个 batch
        for j in range(samples.shape[1]):  # 遍历每个序列位置
            selected_classes[i, j] = np.random.choice(len(charmap), p=samples[i, j])

    decoded_samples = []
    for i in range(len(selected_classes)):
        decoded = []
        for j in range(len(selected_classes[i])):
            decoded.append(inv_charmap[selected_classes[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples

def save(samples):
    with open(args.output, 'a') as f:
        for s in samples:
            #s = "".join(s).replace('`', '')
            f.write(s + "\n")


def generate_other_string(symbol, pcfg):
    if symbol not in pcfg.grammar:
        return symbol
    choices = pcfg.grammar[symbol]
    selected = random.choices(choices, weights=[choice['prob'] for choice in choices], k=1)[0]
    generated_string = random.choices(selected['values'])[0]
    return generated_string

def gen_pcfg(symbol, checkpoint_dic, pcfg, args):
    if symbol[0] == "A":
        checkpoint = checkpoint_dic["A"]
    elif symbol[0] == "D":
        checkpoint = checkpoint_dic["D"]
    elif symbol[0] == "O":
        checkpoint = checkpoint_dic["O"]
    else:
        s = generate_other_string(symbol, pcfg)
        #print(symbol, s)
        return s
    max_len = int(symbol[1])
    with open(Path(os.path.dirname(os.path.dirname(checkpoint)) +"/"+ 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f)

    with open(Path(os.path.dirname(os.path.dirname(checkpoint))  +"/"+ 'inv_charmap.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f)

    # print(len(charmap))
    # print(checkpoint)
    # print(charmap)
    lib.delete_all_params()
    #tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as session2:
            fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
            saver = tf.compat.v1.train.Saver()
            saver.restore(session2, checkpoint) 
            
            samples = generate_samples(session2, fake_inputs, charmap, inv_charmap)            
            _samples = [sample for sample in samples if len("".join(sample).replace('`', '')) == max_len]
            
        if _samples == []:
            return None
    
    return _samples[0]



def sample_run(args, checkpoint_dic, pcfg):
    with open(Path(args.input_dir +"/"+ 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f)

    with open(Path(args.input_dir  +"/"+ 'inv_charmap.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f)
    print(charmap)
    lib.delete_all_params()
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as session:
        fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
        saver = tf.compat.v1.train.Saver()
        saver.restore(session, args.checkpoint)
        samples = []
        
        for i in tqdm(range(int(args.num_samples / args.batch_size))):
            pcfg_samples = generate_samples(session, fake_inputs, charmap, inv_charmap)
            #print(pcfg_samples)
            _samples = []
            for sample in pcfg_samples:
                s = ""
                for symbol in sample:
                    if symbol == "`":
                        break
                    for _ in range(5):
                        _s = gen_pcfg(symbol, checkpoint_dic, pcfg, args)
                        if _s:
                            s += "".join(_s).replace('`', '')
                            break
                if s != "":
                    _samples.append(s)
                
            #print(_samples)
            samples.extend(_samples)
            # append to output file every 1000 batches
            if i % 100 == 0 and i > 0: 
                save(samples)
                samples = [] # flush
                print(f'wrote {100 * args.batch_size} samples to {args.output}. {i * args.batch_size} total.')
        save(samples)
        print(f'finished')

def parse_args():  
    parser = argparse.ArgumentParser(description='Process some integers.')  
  
    # 输入目录  
    parser.add_argument('--input_dir', type=str, default="GAN_pcfg", help='Input directory path')  
  
    # 批处理大小  
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')  
  
    # 序列长度  
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for processing')  
  
    # 层维度  
    parser.add_argument('--layer_dim', type=int, default=128, help='Dimension of layers in the model')  
  
    # 输出文件  
    parser.add_argument('--output', type=str, default="gan_gan.txt", help='Output file path')  
  
    # 样本数量  
    parser.add_argument('--num_samples', type=int, default=1000000, help='Number of samples to generate')  
  
    # 检查点文件  
    parser.add_argument('--checkpoint', type=str, default="GAN_pcfg/checkpoints/checkpoint_95000.ckpt", help='Checkpoint file path')  
  
    # 规则路径  
    parser.add_argument('--rules_path', type=str, default="data/Rules/result", help='Path to the rules directory')  
  
    args = parser.parse_args()  
    return args  

# 使用示例  
if __name__ == "__main__":  
    set_GPU() 
    
    args = parse_args()  
    #print(args)
    pcfg = pcfg_grammer()
    checkpoint_dic = {
        "A": "GAN_A/checkpoints/checkpoint_195000.ckpt",
        "D": "GAN_D/checkpoints/checkpoint_195000.ckpt",
        "O": "GAN_O/checkpoints/checkpoint_195000.ckpt",
    }

    sample_run(args, checkpoint_dic, pcfg)
    

