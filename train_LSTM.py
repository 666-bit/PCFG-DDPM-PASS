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


def parse_args():  
    parser = argparse.ArgumentParser(description='Process some integers.')  
    # 输入目录  
    parser.add_argument('--output_dir', type=str, default="./LSTM", help='Input directory path') 
    
    # 批处理大小  
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')  
  
    # 序列长度  
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for processing')  
  
    # 层维度  
    parser.add_argument('--layer_dim', type=int, default=64, help='Dimension of layers in the model')  
  
    parser.add_argument('--iters', type=int, default=1000000, help='Number of samples to generate')  
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of samples to generate')  
    
    parser.add_argument('--sav_every', '-s',type=int,default=5000,help='Save model checkpoints after this many iterations (default: 5000)')\

    parser.add_argument('--data_set', type=str, default="data/rockyou-train.txt", help='Path to the train data')  
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  
  
    args = parser.parse_args()  
    return args


class SampleGenerationCallback(Callback):
    def __init__(self, args):
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        if (epoch) % self.args.sav_every == 0 and epoch != 0:
            print(f"\n----- Save model after Epoch: {epoch}")
            self.model.save(f"{self.args.output_dir}/checkpoints/model_{self.args.batch_size}_{self.args.epochs}_epoch_{epoch}.h5")
            
def train_run(args):
    lines, charmap, inv_charmap = utils.load_dataset(
        path=args.data_set,
        max_length=args.seq_length,
    )
    startChar = "<BEG>"
    endChar = "<END>"
    lines = [[startChar] + [c for c in line if c != "`"] + [endChar] for line in lines]
    charmap[startChar] = len(inv_charmap)
    inv_charmap.append(startChar)
    charmap[endChar] = len(inv_charmap)
    inv_charmap.append(endChar)
    
    
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.output_dir +"/"+ 'checkpoints').mkdir(exist_ok=True)
    
    with open(args.output_dir +"/"+'charmap.pickle', 'wb') as pkl_file:
        pickle.dump(charmap, pkl_file)
    with open(args.output_dir +"/"+'inv_charmap.pickle', 'wb') as pkl_file:
        pickle.dump(inv_charmap, pkl_file)
        
    max_sequence_len = args.seq_length + 1
    total_words = len(charmap)
    
    def inf_train_gen():
        while True:
            np.random.shuffle(lines)
            for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
                
                data = lines[i:i+args.batch_size]
                input_sequences = []
                for line in data:
                    token_list = [charmap[c] for c in line]
                    for i in range(1, len(token_list)):
                        n_gram_sequence = token_list[:i+1]
                        input_sequences.append(n_gram_sequence)
                
                xs = [line[:-1] for line in input_sequences]
                labels = [line[-1] for line in input_sequences]

                
                #max_sequence_len = max([len(x) for x in input_sequences])
                xs = np.array(pad_sequences(xs, maxlen=max_sequence_len - 1, padding='post', value = charmap["unk"]))

                ys = tf.keras.utils.to_categorical(labels, num_classes=len(charmap))
                
                yield xs, ys
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',  # 监控的指标
        factor=0.8,  # 减少学习率的因子，当指标不再改善时，学习率将以这个因子乘以当前学习率
        patience=10,  # 当指标在若干个epoch内不再改善时减少学习率
        verbose=1,
        mode='min',  # 'min' 代表指标越小越好，'max' 代表指标越大越好
        min_lr=0.0001  # 学习率的下限
    )
    gen = inf_train_gen()
    
    model = Sequential()
    model.add(Embedding(total_words, args.layer_dim, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(args.layer_dim, kernel_regularizer=l2(0.0001))))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate), metrics=['accuracy'])

    for i in tqdm(range(args.iters)):
        xs, ys = next(gen)
        history = model.fit(xs, ys, batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks = [reduce_lr, SampleGenerationCallback(args=args)])
        if (i) % args.sav_every == 0 and i != 0:
            print(f"\n----- Save model after iters: {i}")
            model.save(f"{args.output_dir}/checkpoints/model_{args.batch_size}_{args.iters}_iter_{i}.h5")
        
    model.save(f"{args.output_dir}/checkpoints/model_final.h5")

    
# 使用示例  
if __name__ == "__main__":  
    set_GPU() 
    
    args = parse_args()  

    train_run(args)

    
    

