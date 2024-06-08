import argparse

def read_passwords(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = []
        for line in f:
            data.append(line.strip())
        return data

def main(data_file, rockyou_file):
    # 读取并计算data.txt中的不重复密码
    data_passwords = read_passwords(data_file)
    #data_passwords= data_passwords[:300000]
    unique_passwords = set(data_passwords)
    unique_passwords_count = len(unique_passwords)
    
    # 读取rockyou-test.txt中的密码
    rockyou_passwords = read_passwords(rockyou_file)
    unique_rockyou_passwords = set(rockyou_passwords)
    
    # 计算命中的密码数
    matching_passwords_count = len(unique_passwords.intersection(unique_rockyou_passwords))
    
    # 计算百分比，保留两位小数
    matching_percentage = (matching_passwords_count / len(unique_rockyou_passwords) * 100)
    matching_percentage = round(matching_percentage, 2)
    
    # 输出结果
    print(f"passwords in {data_file}: {len(data_passwords)}")
    print(f"Unique passwords in {data_file}: {unique_passwords_count}")
    print(f"Passwords from {data_file} that are in {rockyou_file}: {matching_passwords_count}")
    print(f"Matching percentage: {matching_percentage}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process password files.")
    parser.add_argument("--data", default="",type=str, help="Path to the data.txt file")
    parser.add_argument("--rockyou", default="data/rockyou-test.txt",type=str, help="Path to the rockyou-test.txt file")
    
    args = parser.parse_args()
    
    main(args.data, args.rockyou)
