import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

def ResBlock(name, inputs, dim):
    output = inputs
    output = tf.nn.relu(output)
    # print(name+'.1', dim, dim, 5, output)
    output = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, seq_len, layer_dim, output_dim, prev_outputs=None):
    # 生成随机噪声，形状为[n_samples, 128]
    output = make_noise(shape=[n_samples, 128])
    # 线性变换，将形状变为[n_samples, seq_len * layer_dim]
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * layer_dim, output)
    # 调整形状为[n_samples, layer_dim, seq_len]
    output = tf.reshape(output, [-1, layer_dim, seq_len])
    # 一系列残差块，每个残差块保持输入输出维度相同
    output = ResBlock('Generator.1', output, layer_dim)
    output = ResBlock('Generator.2', output, layer_dim)
    output = ResBlock('Generator.3', output, layer_dim)
    output = ResBlock('Generator.4', output, layer_dim)
    output = ResBlock('Generator.5', output, layer_dim)
    # 最后一层1D卷积，将形状变为[n_samples, output_dim, seq_len]
    output = lib.ops.conv1d.Conv1D('Generator.Output', layer_dim, output_dim, 1, output)
    # 转置输出维度，形状变为[n_samples, seq_len, output_dim]
    output = tf.transpose(output, [0, 2, 1])
    #print(output.shape)
    # 应用softmax激活函数，形状保持不变
    output = softmax(output, output_dim)
    return output

def Discriminator(inputs, seq_len, layer_dim, input_dim):
    # inputs: 形状为 [n_samples, seq_len, input_dim]
    # input_dim = output_dim = len(charmap)
    # 转置输入的维度，形状变为[n_samples, input_dim, seq_len]
    output = tf.transpose(inputs, [0,2,1])
    # 第一个1D卷积层，将形状变为[n_samples, layer_dim, seq_len] layer_dim = 隐藏层
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', input_dim, layer_dim, 1, output)
    # 一系列残差块，每个残差块保持输入输出维度相同
    output = ResBlock('Discriminator.1', output, layer_dim)
    output = ResBlock('Discriminator.2', output, layer_dim)
    output = ResBlock('Discriminator.3', output, layer_dim)
    output = ResBlock('Discriminator.4', output, layer_dim)
    output = ResBlock('Discriminator.5', output, layer_dim)
    # 调整输出形状为扁平化向量，形状变为[n_samples, seq_len * layer_dim]
    output = tf.reshape(output, [-1, seq_len * layer_dim])
    # 最后一层线性变换，输出一个值，形状为[n_samples, 1]
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * layer_dim, 1, output)
    return output

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random.normal(shape)
