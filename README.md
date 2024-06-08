# 基于PCFG，GAN，DDPM的口令猜测

本项目基于PassGAN改进而来，并使用了pcfg_cracker工具，支持LSTM模型，DDPM模型，与PCFG-GAN模型进行口令生成任务。

## PassGAN baseline
需要tensroflow==2.12.0，cuda11.8 cudnn8.6.0
对PassGAN尽心些许改进，使用WGAN-div作为附加损失函数。


```
#训练
python train_gan.py train --output gan_baseline --training-data data/rockyou-train.txt
#生成
python train_gan.py sample \
	--input-dir gan_baseline \
	--checkpoint gan_baseline/checkpoints/checkpoint_195000.ckpt \
	--output gan_baseline.txt \
	--batch-size 64 \
	--num-samples 1000000

```


## LSTM baseline
需要tensroflow==2.12.0，cuda11.8 cudnn8.6.0
```
#训练
python train_LSTM.py \
	--output_dir LSTM_baseline \
	--data_set data/rockyou-train.txt \
	--iters 1000000 \
	--epochs 100
#生成
python sample_lstm_baseline.py  \
	--input-dir LSTM_baseline \
	--checkpoint LSTM_baseline/checkpoints/model_64_100_epoch5000.h5 \
	--output LSTM_baseline.txt \
	--batch-size 64 \
	--num-samples 1000000

```


## DDPM
与其余模型不同，需要pytorch与cuda11.8
```
#训练
python train_diffusion.py --output_dir diffusion --data_set data/rockyou-train.txt
#生成
python sample_diffusion.py  \
	--input-dir diffusion \
	--checkpoint diffusion/checkpoints/ddpm_fcnet.pth \
	--output diffusion.txt \
	--batch-size 64 \
	--num-samples 1000000


```

## PCFG-GAN
需要tensroflow==2.12.0，cuda11.8 cudnn8.6.0
PCFG-GAN分为两步训练，首先训练noise到PCFG文法的概率，之后训练PCFG文法到具体文本的概率。两步模型使用GAN。这两步的训练都需要PCFG文法数据。

首先，生成PCFG文法数据，依赖pcfg_cracker项目
```
git clone https://github.com/lakiw/pcfg_cracker.git
pip  install chardet
cd ./pcfg_cracker
python trainer.py -t ../data/rockyou-train.txt -r result
#之后把Rules文件夹复制到../data目录下
cp -rp Rules ../data 
#生成数据
cd ../
python gen_PCFG_data.py --output data/ --rule_path data/Rules/result/
python gen_PCFG_data.py --data_type A --output data/ --rule_path data/Rules/result/
python gen_PCFG_data.py --data_type D --output data/ --rule_path data/Rules/result/
python gen_PCFG_data.py --data_type O --output data/ --rule_path data/Rules/result/

```

训练与生成
```
#训练，PCFG，A，D，O四个模型均需要训练，其余标签生成采用传统PCFG方式
python train_PCFG_gan.py train --output gan_PCFG --training-data data/PCFG_train.txt
python train_gan.py train --output gan_A --training-data data/A_train.txt
python train_gan.py train --output gan_D --training-data data/D_train.txt
python train_gan.py train --output gan_O --training-data data/O_train.txt


#生成，需要在checkpoint_dic设置A，D，O三个模型的路径
#如下
checkpoint_dic = {
    "A": "gan_A/checkpoints/checkpoint_195000.ckpt",
    "D": "gan_D/checkpoints/checkpoint_195000.ckpt",
    "O": "gam_O/checkpoints/checkpoint_195000.ckpt",
}

python sample_gan_gan.py --input_dir gan_PCFG --checkpoint gan_PCFG/checkpoints/checkpoint_95000.ckpt --rules_path data/Rules/result --output gan_gan.txt --num_samples 1000000

```


## 评估
与rock-test.txt（约500万条）比较，对比密码碰撞的概率与不重复密码数量

```
python eval --data gan_baseline.txt --rockyou data/rockyou-test.txt

```