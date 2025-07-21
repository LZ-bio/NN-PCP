#from keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K

#def focal_loss1(alpha=0.75, gamma=2.0):
#    def focal_loss_fixed(y_true, y_pred):
        # y_true 是个一阶向量, 下式按照加号分为左右两部分
        # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码”
        # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha
        # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha
#        ones = K.ones_like(y_true)
#        alpha_t = y_true*alpha + (ones-y_true)*(1-alpha)

        # 类似上面，y_true仍然视为 0/1 掩码
        # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值
        # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值
        # 第3部分 K.epsilon() 避免后面 log(0) 溢出
#        p_t = y_true*y_pred + (ones-y_true)*(ones-y_pred) + K.epsilon()

        # 就是公式的字面意思
#        focal_loss = -alpha_t * K.pow((ones-p_t),gamma) * K.log(p_t)
#    return focal_loss_fixed

def focal_loss(alpha=0.75, gamma=3.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred,epsilon,1.0-epsilon)
        y_true = tf.cast(y_true,tf.float32)
        alpha_t = y_true*alpha + (1-y_true)*(1-alpha)
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
        focal_loss = -alpha_t * K.pow((1-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return focal_loss_fixed

# taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))