from __future__ import print_function
import sys
# Check for python3 and error out if not
if sys.version_info[0] < 3:
    print("This program requires Python 3.x", file=sys.stderr)
    sys.exit(1)

import argparse
import os

# Local imports

from pcfg_cracker.lib_guesser.pcfg_grammar import PcfgGrammar

import random
from tqdm import tqdm
import re

def gen(args):

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
    print(base_directory)

    pcfg = PcfgGrammar(
                program_info['rule_name'],
                base_directory,
                program_info['version'],
                base_structure_folder = "Prince",
                skip_case = program_info['skip_case'],
                )

    if args.data_type == "PCFG":
        def read_pcfg_rules(filename):
            rules = []
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    #print(line.split("\t"))
                    
                    rule, probability = line.split("\t")
                    formatted_rule = re.findall(r'([A-Za-z])[0-9]?', rule)
                    #print(formatted_rule)
                    #print()
                    rules.append(("".join(formatted_rule), float(probability)))   
            
            rules_dic = {}
            for rule, pro in rules:
                if rule in rules_dic:
                    rules_dic[rule] += pro
                else:
                    rules_dic[rule] = pro
            return rules_dic

        # 计算每个规则的出现次数
        def calculate_occurrences(rules, total_count):
            occurrences = []
            for rule, probability in rules.items():
                count = int(probability * total_count)
                occurrences.append((rule, count))
            return occurrences

        # 生成数据集
        def generate_dataset(occurrences):
            dataset = []
            for rule, count in occurrences:
                dataset.extend([rule] * count)
            random.shuffle(dataset)
            return dataset

        # 保存数据集到文件
        def save_dataset(dataset, filename):
            with open(filename, 'w') as file:
                for data in dataset:
                    file.write(data + '\n')
        
        input_filename = args.rule_path + 'Grammar/raw_grammar.txt'
        output_filename = args.output + 'PCFG_train.txt'
        total_count = args.num_samples

        rules = read_pcfg_rules(input_filename)
        occurrences = calculate_occurrences(rules, total_count)
        dataset = generate_dataset(occurrences)
        save_dataset(dataset, output_filename)
        return
    
    
    
    allowed_prefixes = {args.data_type}
    base_rules = [rule for rule in pcfg.base if rule['replacements'][0][0] in allowed_prefixes]
    def generate_string(symbol):
        if symbol not in pcfg.grammar:
            return symbol
        choices = pcfg.grammar[symbol]
        selected = random.choices(choices, weights=[choice['prob'] for choice in choices], k=1)[0]
        generated_string = random.choices(selected['values'])[0]
        return generated_string

    # 计算每个基础规则生成的字符串的概率并生成字符串
    def generate_strings_with_probabilities(total_count):
        data = []
        for _ in tqdm(range(total_count)):
            selected = random.choices(base_rules, weights=[rule['prob'] for rule in base_rules], k=1)[0]
            string = generate_string(selected['replacements'][0])
            data.append(string)
            
        return data

    data = generate_strings_with_probabilities(args.num_samples)

    with open(args.output + args.data_type + "_train.txt", 'w') as file:
        for line in data:
            file.write(line + '\n')

def parse_args():  
    parser = argparse.ArgumentParser(description='Process some integers.')  
    # 输入目录  
    parser.add_argument('--data_type', type=str, default="PCFG", help='PCFG, A, O, D')  
    
    parser.add_argument('--rule_path', type=str, default="data/Rules/result/", help='rule path')  
    
    parser.add_argument('--output', type=str, default="data/", help='Output file path')  
  
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples to generate')  
   
  
    args = parser.parse_args()  
    return args  

# 使用示例  
if __name__ == "__main__":  
    
    args = parse_args()  

    gen(args)