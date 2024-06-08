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
from tensorflow import keras
from tensorflow.keras.models import load_model


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

def save(path, samples):
    with open(path, 'a') as f:
        for s in samples:
            #s = "".join(s).replace('`', '')
            f.write(s + "\n")



def sample_run(args):
    with open(Path(args.input_dir +"/"+ 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f)

    with open(Path(args.input_dir  +"/"+ 'inv_charmap.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f)
        
    model = keras.models.load_model(args.checkpoint)
    
    print(charmap)
    
    samples = []
    for i in tqdm(range(args.num_samples)):
        
        seed_text = ["<BEG>"]
        next_words = args.seq_length
        max_sequence_len = args.seq_length + 1
        for _ in range(next_words):
            
            token_list = [charmap[c] for c in seed_text]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='post')
            predicted = model.predict(token_list, verbose=0)
            #print(predicted)
            predicted2 = np.random.choice(range(len(predicted.reshape(-1))), p = predicted.reshape(-1))
            #这个循环是将词典的索引对的索引与预测的标签进行匹配，找到预测的单词索引后，将该单词加入到句子后
            #print(predicted2)
            output_word = inv_charmap[int(predicted2)]
            if output_word == "<END>" or output_word == "unk":
                break
            seed_text = seed_text + [output_word]
            

        password = "".join(seed_text[1:])
        #print(password)
        if password != "":
            samples.append(password)
        # append to output file every 1000 batches
        if i % 1000 == 0 and i > 0: 
            save(args.output, samples)
            samples = [] # flush
            print(f'wrote {i} samples to {args.output}. {args.num_samples} total.')
                
    save(args.output, samples)
    print(f'finished')

def parse_args():  
    parser = argparse.ArgumentParser(description='Process some integers.')  
    # 输入目录  
    parser.add_argument('--input_dir', type=str, default="LSTM", help='Input directory path')  
    
    # 批处理大小  
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')  
  
    # 序列长度  
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for processing')  
  
    # 层维度  
    parser.add_argument('--layer_dim', type=int, default=128, help='Dimension of layers in the model')  
  
    # 输出文件  
    parser.add_argument('--output', type=str, default="lstm_baseline.txt", help='Output file path')  
  
    # 样本数量  
    parser.add_argument('--num_samples', type=int, default=1000000, help='Number of samples to generate')  
  
    # 检查点文件  
    parser.add_argument('--checkpoint', type=str, default="LSTM/checkpoints/model_64_100_epoch5000.h5", help='Checkpoint file path')  
  
    # 规则路径  
  
    args = parser.parse_args()  
    return args  

# 使用示例  
if __name__ == "__main__":  
    set_GPU() 
    
    args = parse_args()  

    sample_run(args)
    
