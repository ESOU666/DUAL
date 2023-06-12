from __future__ import division
import argparse

parser = argparse.ArgumentParser()
# Basic & Model parameters
parser.add_argument("-gpu", type=str, default="0")  # GPU编号，默认为0
parser.add_argument("-message", type=str, default="")  # 附加消息，默认为空字符串
parser.add_argument("-loglv", type=str, default="debug")  # 日志级别，默认为debug
parser.add_argument("-method", type=str, default="DNN")  # 方法名称，默认为DNN
parser.add_argument("-dataset", type=str, default="Books")  # 数据集名称，默认为Books
parser.add_argument("-seed", type=int, default=12345)  # 随机种子，默认为12345
parser.add_argument("-model_path", type=str, default=None)  # 模型路径，默认为None
parser.add_argument("-dims", type=str, default="200,80")  # 维度，默认为"200,80"
parser.add_argument("-optimizer", type=str, default="Adam")  # 优化器名称，默认为Adam
parser.add_argument("-batch_size", type=int, default=64)  # 批量大小，默认为64
parser.add_argument("-maxlen", type=int, default=100)  # 序列最大长度，默认为100
parser.add_argument("-lr", type=float, default=1e-3)  # 学习率，默认为0.001
parser.add_argument("-lr_decay", type=float, default=0.5)  # 学习率衰减，默认为0.5
parser.add_argument("-momentum", type=float, default=0.9)  # 动量，默认为0.9
parser.add_argument("-epochs", type=int, default=2)  # 训练轮数，默认为2
parser.add_argument("-log_every", type=int, default=100)  # 每隔多少步打印训练指标，默认为100
parser.add_argument("-viz_every", type=int, default=200)  # 每隔多少步进行测试和可视化，默认为200
parser.add_argument("-model_every", type=int, default=1e10)  # 每隔多少步保存模型，默认为1e10
parser.add_argument("-output", type=str, default="output")  # 输出目录，默认为"output"
# SVGP parameters
parser.add_argument("-n_ind", type=int, default=200)  # 指数的数量，默认为200
parser.add_argument("-lengthscale", type=float, default=2.0)  # 长度尺度，默认为2.0
parser.add_argument("-amplitude", type=float, default=0.3)  # 振幅，默认为0.3
parser.add_argument("-diag_cov", dest="diag_cov", action="store_true")  # 对角协方差矩阵，默认为False
parser.set_defaults(diag_cov=False)
parser.add_argument("-jitter", type=float, default=1e-4)  # 抖动，默认为0.0001
parser.add_argument("-prior_mean", type=float, default=0.0)  # 先验均值，默认为0.0
parser.add_argument("-temp", type=float, default=1e-6)  # 温度，默认为0.000001
parser.add_argument("-km_coeff", type=float, default=0.1)  # K-Means系数，默认为0.1
args = parser.parse_args()

# ----------------------------------------------------------------

import os

# set environment variables
os.environ["PYTHONHASHSEED"] = str(args.seed)  # 设置Python的hash种子
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # 设置CUDA可见的GPU设备
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 设置TensorFlow的日志级别为3（只显示错误信息）

import random
import shutil
import glob
import numpy as np
import tensorflow as tf

import models
import utils
import dataset

# ----------------------------------------------------------------
# Arguments and Settings

# random seeds
random.seed(args.seed)  # 设置随机数种子
np.random.seed(args.seed)  # 设置NumPy的随机数种子
tf.set_random_seed(args.seed)  # 设置TensorFlow的随机数种子

# copy python files for reproducibility
logger, dirname = utils.setup_logging(args)  # 设置日志
files_to_copy = glob.glob(os.path.dirname(os.path.realpath(__file__)) + "/*.py")  # 获取当前文件夹下所有Python文件的路径
for file_ in files_to_copy:
    script_src = file_
    script_dst = os.path.abspath(os.path.join(dirname, os.path.basename(file_)))  # 拷贝文件到指定目录
    shutil.copyfile(script_src, script_dst)
logger.debug("Copied {} to {} for reproducibility.".format(", ".join(map(os.path.basename, files_to_copy)), dirname))

# global constants & variables
EMBEDDING_DIM = 18  # 嵌入维度
HIDDEN_SIZE = 18 * 2  # 隐藏层大小
ATTENTION_SIZE = 18 * 2  # 注意力大小
best_auc = 0.0  # 最佳AUC

# print arguments
for k, v in sorted(vars(args).items()):  # 打印命令行参数
    logger.info("  %20s: %s" % (k, v))

# get arguments
method = args.method  # 方法名称
n_epochs = args.epochs  # 训练轮数
batch_size = args.batch_size  # 批量大小
maxlen = args.maxlen  # 序列最大长度
lr = args.lr  # 学习率
lr_decay = args.lr_decay  # 学习率衰减
momentum = args.momentum  # 动量
layer_dims = utils.get_ints(args.dims)  # 解析维度字符串为整数列表
temp = args.temp  # 温度

gp_params = dict()
gp_params["num_inducing"] = args.n_ind  # 指数的数量
gp_params["lengthscale"] = args.lengthscale  # 长度尺度
gp_params["amplitude"] = args.amplitude  # 振幅
gp_params["jitter"] = args.jitter  # 抖动
gp_params["n_gh_samples"] = 20  # 高斯-埃尔米特积分点的数量
gp_params["n_mc_samples"] = 2000  # 蒙特卡洛采样点的数量
gp_params["prior_mean"] = args.prior_mean  # 先验均值
gp_params["diag_cov"] = args.diag_cov  # 是否使用对角协方差矩阵
gp_params["km_coeff"] = args.km_coeff  # K-Means系数

# ----------------------------------------------------------------
# Dataset

logger.info("Dataset {} loading...".format(args.dataset))

if args.dataset in ["Books", "Electronics"]:
    data_path = "data/" + args.dataset  # 数据集路径
else:
    logger.error("Invalid dataset : {}".format(args.dataset))
    raise ValueError("Invalid dataset : {}".format(args.dataset))

train_data = dataset.DataIterator("local_train_splitByUser", data_path, batch_size, maxlen)  # 加载训练数据
test_data = dataset.DataIterator("local_test_splitByUser", data_path, batch_size, maxlen)  # 加载测试数据
n_uid, n_mid, n_cat = train_data.get_n()  # 获取用户、物品和类别的数量

logger.info("Dataset {} loaded.".format(args.dataset))
logger.info("# UID: {}, # MID: {}, # CAT: {}.".format(n_uid, n_mid, n_cat))


# helper function for converting data to dense vectors
def prepare_data(input, target, maxlen=None, return_neg=False):
    lengths_x = [len(inp[4]) for inp in input]  # 获取输入序列的长度
    seqs_mid = [inp[3] for inp in input]  # 获取输入序列的物品ID
    seqs_cat = [inp[4] for inp in input]  # 获取输入序列的类别ID
    noclk_seqs_mid = [inp[5] for inp in input]  # 获取未点击序列的物品ID
    noclk_seqs_cat = [inp[6] for inp in input]  # 获取未点击序列的类别ID

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen :])  # 截断输入序列的物品ID
                new_seqs_cat.append(inp[4][l_x - maxlen :])  # 截断输入序列的类别ID
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen :])  # 截断未点击序列的物品ID
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen :])  # 截断未点击序列的类别ID
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = np.zeros((n_samples, maxlen_x)).astype("int64")  # 输入序列的物品ID矩阵
    cat_his = np.zeros((n_samples, maxlen_x)).astype("int64")  # 输入序列的类别ID矩阵
    noclk_mid_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype("int64")  # 未点击序列的物品ID矩阵
    noclk_cat_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype("int64")  # 未点击序列的类别ID矩阵
    mid_mask = np.zeros((n_samples, maxlen_x)).astype("float32")  # 输入序列的掩码矩阵
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, : lengths_x[idx]] = 1.0
        mid_his[idx, : lengths_x[idx]] = s_x
        cat_his[idx, : lengths_x[idx]] = s_y
        noclk_mid_his[idx, : lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, : lengths_x[idx], :] = no_sy

    uids = np.array([inp[0] for inp in input])  # 用户ID数组
    mids = np.array([inp[1] for inp in input])  # 物品ID数组
    cats = np.array([inp[2] for inp in input])  # 类别ID数组

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x), noclk_mid_his, noclk_cat_his
    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)


# ----------------------------------------------------------------
# Model setup

# base models
if method == "DNN":
    model = models.Model_DNN(
        n_uid, n_mid, n_cat, EMBEDDING_DIM, layer_dims=layer_dims, optm=args.optimizer, beta1=momentum, gp_params_dict=gp_params
    )
elif method == "PNN":
    model = models.Model_PNN(
        n_uid, n_mid, n_cat, EMBEDDING_DIM, layer_dims=layer_dims, optm=args.optimizer, beta1=momentum, gp_params_dict=gp_params
    )
elif method == "Wide":
    model = models.Model_WideDeep(
        n_uid, n_mid, n_cat, EMBEDDING_DIM, layer_dims=layer_dims, optm=args.optimizer, beta1=momentum, gp_params_dict=gp_params
    )
elif method == "DIN":
    model = models.Model_DIN(
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        ATTENTION_SIZE,
        layer_dims=layer_dims,
        optm=args.optimizer,
        beta1=momentum,
        gp_params_dict=gp_params,
    )
elif method == "DIEN":
    model = models.Model_DIEN(
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        HIDDEN_SIZE,
        ATTENTION_SIZE,
        layer_dims=layer_dims,
        optm=args.optimizer,
        beta1=momentum,
        gp_params_dict=gp_params,
    )
else:
    logger.error("Invalid method : {}".format(method))
    raise ValueError("Invalid method : {}".format(method))

# ----------------------------------------------------------------
# Training

GPU_OPTIONS = tf.GPUOptions(allow_growth=True)
CONFIG = tf.ConfigProto(gpu_options=GPU_OPTIONS)
sess = tf.Session(config=CONFIG)
global_init_op = tf.global_variables_initializer()
sess.run(global_init_op)

writer = tf.summary.FileWriter(dirname + "/summary/")


def evaluate(sess, test_data, model):
    test_loss_sum = 0.0
    test_accuracy_sum = 0.0
    test_aux_loss_sum = 0.0
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        prob, loss, acc, aux_loss = model.calculate(
            sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, temp]
        )

        test_loss_sum += loss
        test_accuracy_sum += acc
        test_aux_loss_sum = aux_loss
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    test_auc = utils.calc_auc(stored_arr)  # 计算AUC
    test_loss_avg = test_loss_sum / nums  # 平均测试损失
    test_accuracy_avg = test_accuracy_sum / nums  # 平均测试准确率
    test_aux_loss_avg = test_aux_loss_sum / nums  # 平均测试辅助损失
    return test_auc, test_loss_avg, test_accuracy_avg, test_aux_loss_avg


# helper function for adding summary (only simple_value supported)
def write_summary(_writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])  # 创建summary
    _writer.add_summary(summary, step)  # 写入summary
    _writer.flush()


saver = tf.train.Saver(max_to_keep=None)  # 创建Saver对象
if args.model_path:
    saver.restore(sess, args.model_path)  # 从指定路径加载模型
    logger.info("Loaded model from {}".format(args.model_path))

# print variables
logger.debug("Model Variables:")
for p in tf.trainable_variables():
    logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))  # 打印模型变量的名称和形状

# start training
step = 0
loss_list, acc_list, aux_loss_list = [], [], []
test_auc_list, test_loss_list, test_acc_list, test_aux_loss_list = [], [], [], []
for epoch in range(n_epochs):
    for src, tgt in train_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
            src, tgt, maxlen, return_neg=True
        )
        loss, acc, aux_loss, smr = model.train(
            sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats, temp]
        )

        step += 1
        loss_list.append(loss)
        acc_list.append(acc)
        aux_loss_list.append(aux_loss)

        # print training metrics
        if step % args.log_every == 0:
            logger.info(
                "step: {:11d}: train_loss = {:.5f}, train_accuracy = {:.5f}, train_aux_loss = {:.5f}".format(
                    step,
                    np.mean(loss_list[-args.log_every :]),
                    np.mean(acc_list[-args.log_every :]),
                    np.mean(aux_loss_list[-args.log_every :]),
                )
            )
            writer.add_summary(smr, step)
            writer.flush()

        # test and visualization
        if step % args.viz_every == 0:
            test_auc, test_loss, test_accuracy, test_aux_loss = evaluate(sess, test_data, model)
            logger.critical(
                "test_auc: {:.5f}:  test_loss = {:.5f},  test_accuracy = {:.5f},  test_aux_loss = {:.5f}".format(
                    test_auc, test_loss, test_accuracy, test_aux_loss
                )
            )
            test_auc_list.append(test_auc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy)
            test_aux_loss_list.append(test_aux_loss)
            write_summary(writer, tag="Test/auc", value=test_auc, step=step)
            write_summary(writer, tag="Test/accuracy", value=test_accuracy, step=step)
            write_summary(writer, tag="Test/loss", value=test_loss, step=step)

            if best_auc < test_auc:
                best_auc = test_auc
                saver.save(sess, dirname + "/best_model/")
                logger.warning("[{}] Saved best model at step: {}, auc = {:.5f}".format(args.message, step, best_auc))

        # save model
        if step % args.model_every == 0:
            saver.save(sess, dirname + "/model/", global_step=step)
            logger.info("Saved model at step: {}".format(step))

    # learning rate decay after each epoch
    logger.debug("Epoch {:3d} finished. Learning rate reduced from {:.4E} to {:.4E}".format(epoch + 1, lr, lr * lr_decay))
    lr *= lr_decay

logger.error("Experiment [{}] finished. Best auc = {:.5f}.".format(args.message, best_auc))
