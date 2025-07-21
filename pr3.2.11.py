#!/usr/bin/env python
# coding: utf-8

# In[5]:

# # Model construction

# In[30]:


from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from keras.regularizers import l2,Regularizer
from keras import Input
from keras.engine import Model,Layer
from keras import backend as K
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
import scipy.stats as ss
import keras
import numpy as np
from keras import regularizers
# from keras import initializations
from keras.initializers import glorot_uniform, Initializer
from keras.layers import activations, initializers, constraints,Reshape
# our layer will take input shape (nb_samples, 1)
from keras.regularizers import Regularizer
import tensorflow as tf
import re
from Mnes import mgsea,imputation,createNetwork1,createNetwork2,cal_mgsea,createNetwork4,createNetwork5
from scipy.spatial.distance import pdist, squareform
from coms import focal_loss,f1


# In[32]:


class M_Nets(Layer):   
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='lecun_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 b_regularizer=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularize = regularizers.get(b_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        
        super(M_Nets, self).__init__(**kwargs)


    def build(self, input_shape):  

        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)  
        self.n_inputs_per_node = input_dimension / self.units

        rows = np.arange(input_dimension) 
        cols = np.arange(self.units)    
        cols = np.repeat(cols, self.n_inputs_per_node) 
        self.nonzero_ind = np.column_stack((rows, cols)) 

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),  
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        trainable=True
                                        
                                       )
        else:
            self.bias = None

        super(M_Nets, self).build(input_shape)  

    def call(self, x, mask=None):
        
        n_features = x.shape[1]


        kernel = K.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        mult = K.reshape(mult, (-1, int(self.n_inputs_per_node)))
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, self.units))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'kernel_initializer' : self.kernel_initializer,
            'bias_initializer' : self.bias_initializer,
            'use_bias': self.use_bias
        }
        base_config = super(M_Nets, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# In[33]:


class Nets(Layer):
    def __init__(self, units, mapp=None, nonzero_ind=None, kernel_initializer='glorot_uniform',
                 activation='elu', use_bias=True,bias_initializer='glorot_uniform', bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        self.units = units
        self.activation = activation
        self.mapp = mapp
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.l2(0.001)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.l2(0.001)
        self.activation_fn = activations.get(activation)
        super(Nets, self).__init__(**kwargs)

        
    def build(self, input_shape):
        
        input_dim = input_shape[1]
   

        if not self.mapp is None:
            self.mapp = self.mapp.astype(np.float32)

   
        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.mapp)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)
        

        nonzero_count = self.nonzero_ind.shape[0]  


        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer
                                        )
        else:
            self.bias = None

        super(Nets, self).build(input_shape)  
      

    def call(self, inputs):
        
        
        temp_t = tf.scatter_nd(tf.constant(self.nonzero_ind, tf.int32), self.kernel_vector,
                           tf.constant(list(self.kernel_shape)))
    
        output = K.dot(inputs, temp_t)
        
    
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
          
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(Nets, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
      
        return (input_shape[0], self.units)



def sequence_masking(x, mask, mode=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12

#Note: MyMultiHeadAttention was downloaded from webserver(https://github.com/ZJJHYM/bert4keras_note/blob/master/examples/multiattention_learn.py)
class MyMultiHeadAttention(Layer):
    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs
                 ):
        self.heads = heads # 多头的数目
        self.head_size = head_size # v的头的大小
        self.out_dim = heads * head_size # 最终的输出维度
        
        # 一般情况下和head_size是一样的 key_size的作用可以参考苏神的低秩分解：https://kexue.fm/archives/7325
        self.key_size = key_size or head_size # qk每个头的大小  
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        super(MyMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyMultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.heads * self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.heads * self.key_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, a_mask=None, **kwargs):
        # 如果是自注意力，那么q,k,v都是同一个，shape=(btz, seq_len, emb_dim)
        # 如果不是自注意力，则q:shape=shape=(btz, q_seq_len, emb_dim), k,v相同:shape=shape=(btz, k_seq_len, emb_dim)
        q, k, v = inputs[:3]
        q_mask, v_mask, n = None, None, 3
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0], K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2], K.floatx())
        if a_mask:
            a_mask = inputs[n]
            n += 1

        # 经过全连接层

        #将最后一维映射成需要的维度
        qw = self.q_dense(q) # shape=(btz, q_seq_len, self.heads * self.key_size)
        kw = self.k_dense(k) # shape=(btz, k_seq_len, self.heads * self.key_size)
        vw = self.v_dense(v) # shape=(btz, k_seq_len, self.heads * self.head_size)  注意v这里和q,k有点不同
        # 形状变换

        #拆成多头
        qw = K.reshape(qw, shape=(-1, K.shape(qw)[1], self.heads, self.key_size))  # shape=(btz, q_seq_len, self.heads， self.key_size)
        kw = K.reshape(kw, shape=(-1, K.shape(kw)[1], self.heads, self.key_size))  # shape=(btz, k_seq_len, self.heads， self.key_size)
        vw = K.reshape(vw, shape=(-1, K.shape(vw)[1], self.heads, self.head_size))  # shape=(btz, k_seq_len, self.heads， self.head_size)
        # 维度置换 方便计算q*k
        qw = K.permute_dimensions(qw, (0, 2, 1, 3)) # shape=(btz, self.heads，q_seq_len, self.key_size)
        kw = K.permute_dimensions(kw, (0, 2, 1, 3)) # shape=(btz, self.heads，k_seq_len, self.key_size)
        vw = K.permute_dimensions(vw, (0, 2, 1, 3)) # shape=(btz, self.heads，k_seq_len, self.head_size)
        # attention  关于batch_dot的运算方式各位可以看看官方解释，三维好理解，四维也还可以，不过这里的用法会明显不对，计算结束后悔变成shape=(btz, self.heads，q_seq_len, self.heads，k_seq_len)五维
        # a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5 # shape=(btz, self.heads，q_seq_len，k_seq_len)

        # 所以个人比较喜欢用tf.enisum 只需要知道矩阵运算后的shape即可
        # 其中b:btz, h:self.heads, j:q_seq_len, k:k_seq_len, d:self.key_size
        a = tf.einsum('bhjd,bhkd->bhjk', qw, kw) / self.key_size ** 0.5 # shape=(btz, self.heads，q_seq_len，k_seq_len)

        # 形状变换，方便mask计算，因为mask的通用形状一般为(btz, seq_len)，a需要对齐
        a = K.permute_dimensions(a, (0, 3, 2, 1)) # shape=(btz, q_seq_len, self.heads, k_seq_len)
        a = sequence_masking(a, v_mask, 1, -1)
        a = K.permute_dimensions(a, (0, 3, 2, 1)) # shape=(btz, self.heads，q_seq_len，k_seq_len)
        if a_mask is not None:
            a = a - (1 - a_mask) * 1e12

        a = K.softmax(a) # shape=(btz, self.heads，q_seq_len，k_seq_len)

        # a的shape=(btz, self.heads，q_seq_len，k_seq_len);
        # vw的shape=(btz, k_seq_len, self.heads， self.head_size)
        # o = K.batch_dot(a, vw, [3, 2]) # shape=(btz, self.heads，q_seq_len，self.head_size)
        o = tf.einsum('bhjk,bkhd->bhjd', a, vw) # shape=(btz, self.heads，q_seq_len，self.head_size)
        o = K.permute_dimensions(o, (0, 2, 1, 3)) # shape=(btz, q_seq_len, self.heads,self.head_size)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim)) # shape=(btz, q_seq_len, self.out_dim) 因为self.out_dim=self.heads,self.head_size
        if self.key_size != self.head_size:
            o = Dense(
                units=self.head_size,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
        o = sequence_masking(o, q_mask, 0)
        return o # shape=(btz, q_seq_len, self.out_dim)
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class ZeroRowFillLayer(Layer):
    def __init__(self, **kwargs):
        super(ZeroRowFillLayer, self).__init__(**kwargs)
 
    def call(self, inputs):
        non_zero_mask = tf.reduce_any(tf.not_equal(inputs,0),axis=1)
        non_zero_rows = tf.boolean_mask(inputs,non_zero_mask)
        row_mean = tf.reduce_mean(non_zero_rows,axis=0)
        row_mean = row_mean[tf.newaxis,:]
        row_mean = tf.tile(row_mean, [tf.shape(inputs)[0],1])
        zero_mask = tf.reduce_all(tf.equal(inputs,0),axis=1)
        filled_inputs = tf.where(zero_mask,row_mean,inputs)
 
        return filled_inputs
 
    def compute_output_shape(self, input_shape):
        return input_shape

#no_mask and no_attention
def create_models_pheno10(Omics_data):
    #mrna
    M_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_m')

    m0 = Nets(Get_Node_relation_mrna[0].shape[1],mapp =Get_Node_relation_mrna[0].values, name = 'm0')(M_inputs)
    drop0 = keras.layers.Dropout(0.2)(m0)
    drop0 = BatchNormalization()(drop0)
    
    m1 = Nets(Get_Node_relation_mrna[1].shape[1],mapp =Get_Node_relation_mrna[1].values, name = 'm1')(drop0)
    drop_m1 = keras.layers.Dropout(0.5)(m1)
    drop_m1 = BatchNormalization()(drop_m1)
      
    m2 = Nets(Get_Node_relation_mrna[2].shape[1],mapp =Get_Node_relation_mrna[2].values,name = 'm2')(drop_m1)
    drop_m21 = keras.layers.Dropout(0.5)(m2)
    drop_m21 = BatchNormalization()(drop_m21)
    
    task1 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_m21)
    task11 = keras.layers.Dropout(0.5)(task1)
    task12 = BatchNormalization()(task11)
    task13 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task12)
    task13 = keras.layers.Dropout(0.5)(task13)
    task13 = BatchNormalization()(task13)
    task14 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task13)
    task14 = BatchNormalization()(task14)
    output1 = keras.layers.Dense(1,activation='sigmoid')(task14)
    
    #cnv_amp
    h_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_h')

    h0 = Nets(Get_Node_relation_amp[0].shape[1],mapp =Get_Node_relation_amp[0].values, name = 'h0')(h_inputs)
    drop_h0 = keras.layers.Dropout(0.2)(h0)
    drop_h0 = BatchNormalization()(drop_h0)

    h1 = Nets(Get_Node_relation_amp[1].shape[1],mapp =Get_Node_relation_amp[1].values, name = 'h1')(drop_h0)
    drop_h1 = keras.layers.Dropout(0.5)(h1)
    drop_h1 = BatchNormalization()(drop_h1)

    h2 = Nets(Get_Node_relation_amp[2].shape[1],mapp =Get_Node_relation_amp[2].values,name = 'h2')(drop_h1)
    drop_h21 = keras.layers.Dropout(0.5)(h2)
    drop_h21 = BatchNormalization()(drop_h21)

    task2 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h21)
    task21 = keras.layers.Dropout(0.5)(task2)
    task22 = BatchNormalization()(task21)
    task23 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task22)
    task23 = keras.layers.Dropout(0.5)(task23)
    task23 = BatchNormalization()(task23)
    task24 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task23)
    task24 = BatchNormalization()(task24)
    output2 = keras.layers.Dense(1,activation='sigmoid')(task24)

    #cnv_del
    s_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_s')

    s0 = Nets(Get_Node_relation_del[0].shape[1],mapp =Get_Node_relation_del[0].values, name = 's0')(s_inputs)
    drop_s0 = keras.layers.Dropout(0.2)(s0)
    drop_s0 = BatchNormalization()(drop_s0)

    s1 = Nets(Get_Node_relation_del[1].shape[1],mapp =Get_Node_relation_del[1].values, name = 's1')(drop_s0)
    drop_s1 = keras.layers.Dropout(0.5)(s1)
    drop_s1 = BatchNormalization()(drop_s1)

    s2 = Nets(Get_Node_relation_del[2].shape[1],mapp =Get_Node_relation_del[2].values,name = 's2')(drop_s1)
    drop_s21 = keras.layers.Dropout(0.5)(s2)
    drop_s21 = BatchNormalization()(drop_s21)
    
    task3 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s21)
    task31 = keras.layers.Dropout(0.5)(task3)
    task32 = BatchNormalization()(task31)
    task33 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task32)
    task33 = keras.layers.Dropout(0.5)(task33)
    task33 = BatchNormalization()(task33)
    task34 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task33)
    task34 = BatchNormalization()(task34)
    output3 = keras.layers.Dense(1,activation='sigmoid')(task34)


    ##mrna+cnv_amp
    b1 = keras.layers.concatenate([drop_m21,drop_h21])
    b1 = BatchNormalization()(b1)
    
    b11 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b1)
    drop_b11 = keras.layers.Dropout(0.5)(b11)
    b12 = BatchNormalization()(drop_b11)
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b13 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b12)
    b13 = keras.layers.Dropout(0.5)(b13)
    b13 = BatchNormalization()(b13)
    b14 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b13)
    b14 = BatchNormalization()(b14)

    output5 = keras.layers.Dense(1,activation='sigmoid')(b14)
    
    ##cnv + dmeth
    b2 = keras.layers.concatenate([drop_h21,drop_s21])
    b2 = BatchNormalization()(b2)
    
    b21 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b2)
    drop_b21 = keras.layers.Dropout(0.5)(b21)
    b22 = BatchNormalization()(drop_b21)
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b23 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b22)
    b23 = keras.layers.Dropout(0.5)(b23)
    b23 = BatchNormalization()(b23)
    b24 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b23)
    b24 = BatchNormalization()(b24)

    output6 = keras.layers.Dense(1,activation='sigmoid')(b24)

    ##mrna + dmeth
    b3 = keras.layers.concatenate([drop_h21,drop_s21])
    b3 = BatchNormalization()(b3)
    
    b31 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b3)
    drop_b31 = keras.layers.Dropout(0.5)(b31)
    b32 = BatchNormalization()(drop_b31)

    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b33 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b32)
    b33 = keras.layers.Dropout(0.5)(b33)
    b33 = BatchNormalization()(b33)
    b34 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b33)
    b34 = BatchNormalization()(b34)

    output7 = keras.layers.Dense(1,activation='sigmoid')(b34)
    
    ##

    a1 = keras.layers.concatenate([drop_m21,drop_h21,drop_s21])
    a1 = BatchNormalization()(a1)
    
    a2 = Nets(Get_Node_relation_next.shape[1],mapp =Get_Node_relation_next.values,name = 'a1')(a1)
    drop_a2 = keras.layers.Dropout(0.5)(a2)
    drop_a2 = BatchNormalization()(drop_a2)

    a3 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_a2)
    drop_a3 = keras.layers.Dropout(0.5)(a3)
    drop_a3 = BatchNormalization()(drop_a3)
 
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    a5 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_a3)
    a5 = keras.layers.Dropout(0.5)(a5)
    a5 = BatchNormalization()(a5)
    a6 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(a5)
    a6 = BatchNormalization()(a6)

    Output = keras.layers.Dense(1,activation='sigmoid')(a6)



    model = Model(inputs=[M_inputs,h_inputs,s_inputs], outputs=[output1,output2,output3,output5,output6,output7,Output])

    model.summary()

    #opt = keras.optimizers.Adamax(lr = 0.002)
    opt = keras.optimizers.Adam(lr = 0.0005) #,decay=-0.0001 ,decay=0.001
    model.compile(optimizer=opt,
                  #loss=['binary_crossentropy']*5,
                  loss=[focal_loss(alpha=0.5,gamma=1.5)]*7,  #PRAD(0.7,2)
                  loss_weights = [1,1,1,1,1,1,4],   #[1,1,1,1,40,54,20,2000]
                  metrics=['acc'])
    return model


#no_mask and no_attention
def create_models_pheno11(Omics_data):
    #mrna
    M_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_m')

    #m0 = Nets(Get_Node_relation_mrna[0].shape[1],mapp =Get_Node_relation_mrna[0].values, name = 'm0')(M_inputs)
    m0 = keras.layers.Dense(Get_Node_relation_mrna[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(M_inputs)
    drop0 = keras.layers.Dropout(0.2)(m0)
    drop0 = BatchNormalization()(drop0)
    
    #m1 = keras.layers.Dense(Get_Node_relation_mrna[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop0)
    m1 = Nets(Get_Node_relation_mrna[1].shape[1],mapp =Get_Node_relation_mrna[1].values, name = 'm1')(drop0)
    drop_m1 = keras.layers.Dropout(0.5)(m1)
    drop_m1 = BatchNormalization()(drop_m1)

    #m2 = keras.layers.Dense(Get_Node_relation_mrna[2].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_m1)
    m2 = Nets(Get_Node_relation_mrna[2].shape[1],mapp =Get_Node_relation_mrna[2].values,name = 'm2')(drop_m1)
    drop_m21 = keras.layers.Dropout(0.5)(m2)
    drop_m21 = BatchNormalization()(drop_m21)
    
    task1 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_m21)
    task11 = keras.layers.Dropout(0.5)(task1)
    task12 = BatchNormalization()(task11)
    task13 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task12)
    task13 = keras.layers.Dropout(0.5)(task13)
    task13 = BatchNormalization()(task13)
    task14 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task13)
    task14 = BatchNormalization()(task14)
    output1 = keras.layers.Dense(1,activation='sigmoid')(task14)
    
    #cnv_amp
    h_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_h')

    h0 = keras.layers.Dense(Get_Node_relation_amp[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(h_inputs)
    #h0 = Nets(Get_Node_relation_amp[0].shape[1],mapp =Get_Node_relation_amp[0].values, name = 'h0')(h_inputs)
    drop_h0 = keras.layers.Dropout(0.2)(h0)
    drop_h0 = BatchNormalization()(drop_h0)

    #h1 = keras.layers.Dense(Get_Node_relation_amp[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h0)
    h1 = Nets(Get_Node_relation_amp[1].shape[1],mapp =Get_Node_relation_amp[1].values, name = 'h1')(drop_h0)
    drop_h1 = keras.layers.Dropout(0.5)(h1)
    drop_h1 = BatchNormalization()(drop_h1)

    #h2 = keras.layers.Dense(Get_Node_relation_amp[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h1)
    h2 = Nets(Get_Node_relation_amp[2].shape[1],mapp =Get_Node_relation_amp[2].values,name = 'h2')(drop_h1)
    drop_h21 = keras.layers.Dropout(0.5)(h2)
    drop_h21 = BatchNormalization()(drop_h21)

    task2 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h21)
    task21 = keras.layers.Dropout(0.5)(task2)
    task22 = BatchNormalization()(task21)
    task23 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task22)
    task23 = keras.layers.Dropout(0.5)(task23)
    task23 = BatchNormalization()(task23)
    task24 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task23)
    task24 = BatchNormalization()(task24)
    output2 = keras.layers.Dense(1,activation='sigmoid')(task24)

    #cnv_del
    s_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_s')

    s0 = keras.layers.Dense(Get_Node_relation_del[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001), name = 's0')(s_inputs)
    #s0 = Nets(Get_Node_relation_del[0].shape[1],mapp =Get_Node_relation_del[0].values, name = 's0')(s_inputs)
    drop_s0 = keras.layers.Dropout(0.2)(s0)
    drop_s0 = BatchNormalization()(drop_s0)

    #s1 = keras.layers.Dense(Get_Node_relation_del[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s0)
    s1 = Nets(Get_Node_relation_del[1].shape[1],mapp =Get_Node_relation_del[1].values, name = 's1')(drop_s0)
    drop_s1 = keras.layers.Dropout(0.5)(s1)
    drop_s1 = BatchNormalization()(drop_s1)

    #s2 = keras.layers.Dense(Get_Node_relation_del[2].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s1)
    s2 = Nets(Get_Node_relation_del[2].shape[1],mapp =Get_Node_relation_del[2].values,name = 's2')(drop_s1)
    drop_s21 = keras.layers.Dropout(0.5)(s2)
    drop_s21 = BatchNormalization()(drop_s21)
    
    task3 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s21)
    task31 = keras.layers.Dropout(0.5)(task3)
    task32 = BatchNormalization()(task31)
    task33 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task32)
    task33 = keras.layers.Dropout(0.5)(task33)
    task33 = BatchNormalization()(task33)
    task34 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task33)
    task34 = BatchNormalization()(task34)
    output3 = keras.layers.Dense(1,activation='sigmoid')(task34)


    ##mrna+cnv_amp
    b1 = keras.layers.concatenate([drop_m21,drop_h21])
    b1 = BatchNormalization()(b1)
    
    b11 = keras.layers.Dense(144,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b1)
    drop_b11 = keras.layers.Dropout(0.5)(b11)
    b12 = BatchNormalization()(drop_b11)
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b13 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b12)
    b13 = keras.layers.Dropout(0.5)(b13)
    b13 = BatchNormalization()(b13)
    b14 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b13)
    b14 = BatchNormalization()(b14)

    output5 = keras.layers.Dense(1,activation='sigmoid')(b14)
    
    ##cnv + dmeth
    b2 = keras.layers.concatenate([drop_h21,drop_s21])
    b2 = BatchNormalization()(b2)
    
    b21 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b2)
    drop_b21 = keras.layers.Dropout(0.5)(b21)
    b22 = BatchNormalization()(drop_b21)
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b23 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b22)
    b23 = keras.layers.Dropout(0.5)(b23)
    b23 = BatchNormalization()(b23)
    b24 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b23)
    b24 = BatchNormalization()(b24)

    output6 = keras.layers.Dense(1,activation='sigmoid')(b24)

    ##mrna + dmeth
    b3 = keras.layers.concatenate([drop_h21,drop_s21])
    b3 = BatchNormalization()(b3)
    
    b31 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b3)
    drop_b31 = keras.layers.Dropout(0.5)(b31)
    b32 = BatchNormalization()(drop_b31)

    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b33 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b32)
    b33 = keras.layers.Dropout(0.5)(b33)
    b33 = BatchNormalization()(b33)
    b34 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b33)
    b34 = BatchNormalization()(b34)

    output7 = keras.layers.Dense(1,activation='sigmoid')(b34)
    
    ##

    a1 = keras.layers.concatenate([drop_m21,drop_h21,drop_s21])
    a1 = BatchNormalization()(a1)
    
    #a2 = keras.layers.Dense(Get_Node_relation_next.shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(a1)
    a2 = Nets(Get_Node_relation_next.shape[1],mapp =Get_Node_relation_next.values,name = 'a1')(a1)
    drop_a2 = keras.layers.Dropout(0.5)(a2)
    drop_a2 = BatchNormalization()(drop_a2)

    a3 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_a2)
    drop_a3 = keras.layers.Dropout(0.5)(a3)
    drop_a3 = BatchNormalization()(drop_a3)
 
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    a5 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_a3)
    a5 = keras.layers.Dropout(0.5)(a5)
    a5 = BatchNormalization()(a5)
    a6 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(a5)
    a6 = BatchNormalization()(a6)

    Output = keras.layers.Dense(1,activation='sigmoid')(a6)

    model = Model(inputs=[M_inputs,h_inputs,s_inputs], outputs=[output1,output2,output3,output5,output6,output7,Output])

    model.summary()

    opt = keras.optimizers.Adamax(lr = 0.002)
    #opt = keras.optimizers.Adam(lr = 0.0005) #,decay=-0.0001 ,decay=0.001
    model.compile(optimizer=opt,
                  #loss=['binary_crossentropy']*5,
                  loss=[focal_loss(alpha=0.6,gamma=1.5)]*7, #PRAD(0.7,2.5)
                  loss_weights = [3,3,3,1,1,1,120],   #[1,1,1,1,40,54,20,2000]
                  metrics=['acc'])
    return model


#no_mask and no_attention
def create_models_pheno12(Omics_data):
    #mrna
    M_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_m')

    #m0 = Nets(Get_Node_relation_mrna[0].shape[1],mapp =Get_Node_relation_mrna[0].values, name = 'm0')(M_inputs)
    m0 = keras.layers.Dense(Get_Node_relation_mrna[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(M_inputs)
    drop0 = keras.layers.Dropout(0.2)(m0)
    drop0 = BatchNormalization()(drop0)
    
    m1 = keras.layers.Dense(Get_Node_relation_mrna[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop0)
    #m1 = Nets(Get_Node_relation_mrna[1].shape[1],mapp =Get_Node_relation_mrna[1].values, name = 'm1')(drop0)
    drop_m1 = keras.layers.Dropout(0.5)(m1)
    drop_m1 = BatchNormalization()(drop_m1)

    m2 = keras.layers.Dense(Get_Node_relation_mrna[2].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_m1)
    #m2 = Nets(Get_Node_relation_mrna[2].shape[1],mapp =Get_Node_relation_mrna[2].values,name = 'm2')(drop_m1)
    drop_m21 = keras.layers.Dropout(0.5)(m2)
    drop_m21 = BatchNormalization()(drop_m21)
    
    task1 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_m21)
    task11 = keras.layers.Dropout(0.5)(task1)
    task12 = BatchNormalization()(task11)
    task13 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task12)
    task13 = keras.layers.Dropout(0.5)(task13)
    task13 = BatchNormalization()(task13)
    task14 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task13)
    task14 = BatchNormalization()(task14)
    output1 = keras.layers.Dense(1,activation='sigmoid')(task14)
    
    #cnv_amp
    h_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_h')

    h0 = keras.layers.Dense(Get_Node_relation_amp[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(h_inputs)
    #h0 = Nets(Get_Node_relation_amp[0].shape[1],mapp =Get_Node_relation_amp[0].values, name = 'h0')(h_inputs)
    drop_h0 = keras.layers.Dropout(0.2)(h0)
    drop_h0 = BatchNormalization()(drop_h0)

    h1 = keras.layers.Dense(Get_Node_relation_amp[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h0)
    #h1 = Nets(Get_Node_relation_amp[1].shape[1],mapp =Get_Node_relation_amp[1].values, name = 'h1')(drop_h0)
    drop_h1 = keras.layers.Dropout(0.5)(h1)
    drop_h1 = BatchNormalization()(drop_h1)

    h2 = keras.layers.Dense(Get_Node_relation_amp[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h1)
    #h2 = Nets(Get_Node_relation_amp[2].shape[1],mapp =Get_Node_relation_amp[2].values,name = 'h2')(drop_h1)
    drop_h21 = keras.layers.Dropout(0.5)(h2)
    drop_h21 = BatchNormalization()(drop_h21)

    task2 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_h21)
    task21 = keras.layers.Dropout(0.5)(task2)
    task22 = BatchNormalization()(task21)
    task23 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task22)
    task23 = keras.layers.Dropout(0.5)(task23)
    task23 = BatchNormalization()(task23)
    task24 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task23)
    task24 = BatchNormalization()(task24)
    output2 = keras.layers.Dense(1,activation='sigmoid')(task24)

    #cnv_del
    s_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_s')

    s0 = keras.layers.Dense(Get_Node_relation_del[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001), name = 's0')(s_inputs)
    #s0 = Nets(Get_Node_relation_del[0].shape[1],mapp =Get_Node_relation_del[0].values, name = 's0')(s_inputs)
    drop_s0 = keras.layers.Dropout(0.2)(s0)
    drop_s0 = BatchNormalization()(drop_s0)

    s1 = keras.layers.Dense(Get_Node_relation_del[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s0)
    #s1 = Nets(Get_Node_relation_del[1].shape[1],mapp =Get_Node_relation_del[1].values, name = 's1')(drop_s0)
    drop_s1 = keras.layers.Dropout(0.5)(s1)
    drop_s1 = BatchNormalization()(drop_s1)

    s2 = keras.layers.Dense(Get_Node_relation_del[2].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s1)
    #s2 = Nets(Get_Node_relation_del[2].shape[1],mapp =Get_Node_relation_del[2].values,name = 's2')(drop_s1)
    drop_s21 = keras.layers.Dropout(0.5)(s2)
    drop_s21 = BatchNormalization()(drop_s21)
    
    task3 =  keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_s21)
    task31 = keras.layers.Dropout(0.5)(task3)
    task32 = BatchNormalization()(task31)
    task33 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task32)
    task33 = keras.layers.Dropout(0.5)(task33)
    task33 = BatchNormalization()(task33)
    task34 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(task33)
    task34 = BatchNormalization()(task34)
    output3 = keras.layers.Dense(1,activation='sigmoid')(task34)


    ##mrna+cnv_amp
    b1 = keras.layers.concatenate([drop_m21,drop_h21])
    b1 = BatchNormalization()(b1)
    
    b11 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b1)
    drop_b11 = keras.layers.Dropout(0.5)(b11)
    b12 = BatchNormalization()(drop_b11)
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b13 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b12)
    b13 = keras.layers.Dropout(0.5)(b13)
    b13 = BatchNormalization()(b13)
    b14 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b13)
    b14 = BatchNormalization()(b14)

    output5 = keras.layers.Dense(1,activation='sigmoid')(b14)
    
    ##cnv + dmeth
    b2 = keras.layers.concatenate([drop_h21,drop_s21])
    b2 = BatchNormalization()(b2)
    
    b21 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b2)
    drop_b21 = keras.layers.Dropout(0.5)(b21)
    b22 = BatchNormalization()(drop_b21)
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b23 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b22)
    b23 = keras.layers.Dropout(0.5)(b23)
    b23 = BatchNormalization()(b23)
    b24 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b23)
    b24 = BatchNormalization()(b24)

    output6 = keras.layers.Dense(1,activation='sigmoid')(b24)

    ##mrna + dmeth
    b3 = keras.layers.concatenate([drop_h21,drop_s21])
    b3 = BatchNormalization()(b3)
    
    b31 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b3)
    drop_b31 = keras.layers.Dropout(0.5)(b31)
    b32 = BatchNormalization()(drop_b31)

    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    b33 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b32)
    b33 = keras.layers.Dropout(0.5)(b33)
    b33 = BatchNormalization()(b33)
    b34 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(b33)
    b34 = BatchNormalization()(b34)

    output7 = keras.layers.Dense(1,activation='sigmoid')(b34)
    
    ##

    a1 = keras.layers.concatenate([drop_m21,drop_h21,drop_s21])
    a1 = BatchNormalization()(a1)
    
    a2 = keras.layers.Dense(Get_Node_relation_next.shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(a1)
    #a2 = Nets(Get_Node_relation_next.shape[1],mapp =Get_Node_relation_next.values,name = 'a1')(a1)
    drop_a2 = keras.layers.Dropout(0.5)(a2)
    drop_a2 = BatchNormalization()(drop_a2)

    a3 = keras.layers.Dense(72,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_a2)
    drop_a3 = keras.layers.Dropout(0.5)(a3)
    drop_a3 = BatchNormalization()(drop_a3)
 
    #h6 = Dense(8, activation='relu')(h5)
    #a4 = BatchNormalization()(a4)
    #drop_a4 = keras.layers.Dropout(0.5)(a4)

    a5 = keras.layers.Dense(36,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(drop_a3)
    a5 = keras.layers.Dropout(0.5)(a5)
    a5 = BatchNormalization()(a5)
    a6 = keras.layers.Dense(8,activation='elu',kernel_regularizer=regularizers.l2(0.001),use_bias=True,bias_regularizer=regularizers.l2(0.001))(a5)
    a6 = BatchNormalization()(a6)

    Output = keras.layers.Dense(1,activation='sigmoid')(a6)

    model = Model(inputs=[M_inputs,h_inputs,s_inputs], outputs=[output1,output2,output3,output5,output6,output7,Output])

    model.summary()

    #opt = keras.optimizers.Adamax(lr = 0.002)
    opt = keras.optimizers.Adam(lr = 0.0005) #,decay=-0.0001 ,decay=0.001
    model.compile(optimizer=opt,
                  #loss=['binary_crossentropy']*5,
                  loss=[focal_loss(alpha=0.6,gamma=1.5)]*7, #PRAD(0.7,2.5)
                  loss_weights = [1,1,1,1,1,1,4],   #[1,1,1,1,40,54,20,2000]
                  metrics=['acc'])
    return model


def combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type, use_coding_genes_only=False):    
    cols_list_set = [set(list(c)) for c in cols_list]      
    print('cols_list_set',len(cols_list_set))
    if combine_type == 'intersection':    
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)     
    print('intersection_cols',len(cols))
    if use_coding_genes_only: #true
        coding_genes_df = pd.read_csv('./data/protein-coding_gene_with_coordinate_minimal.txt', sep='\t', header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())     
        cols = cols.intersection(coding_genes)  
        print('protein-coding_genes',len(coding_genes))   
    print('finally_cols',len(cols))   
    all_cols = list(cols)
    all_cols_df = pd.DataFrame(index=all_cols) 
    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how='right')  
        df = df.T
        df = df.fillna(0)
        df_list.append(df)
    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )    
    all_data = all_data.swaplevel(i=0, j=1, axis=1)
    order = all_data.columns.levels[0] 
    all_data = all_data.reindex(columns=order, level=0)  
    x = all_data
    reordering_df = pd.DataFrame(index=all_data.index)  
    y = reordering_df.join(y, how='left')   
    y = y.values   
    cols = all_data.columns   
    rows = all_data.index      
    print(
        'After combining, loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], y.shape[0]))
    return x, y, rows, cols

# In[44]:


import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
   

from sklearn.metrics import precision_recall_curve
def evaluates(y_test, y_pred):
    
    auc = metrics.roc_auc_score(y_test,y_pred)
    
    aupr = average_precision_score(y_test, y_pred)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)    
    auprc  = metrics.auc(recall, precision)
    
    pp = [1 if index>=0.5  else 0 for index in  y_pred ]
    
    pre = metrics.precision_score(y_test,pp)
    
    f1 = metrics.f1_score(y_test,pp)
    
    rec = metrics.recall_score(y_test,pp)
    
    acc = metrics.accuracy_score(y_test,pp)
    
    print(confusion_matrix(y_test,pp))
    
    return pre,acc,rec,f1,auc,aupr,auprc


# In[37]:


from deepexplain.model_utils import get_layers, get_coef_importance

def get_coef_importances(model, X_train, y_train, target=-1, feature_importance='deepexplain_grad*input'):

    coef_ = get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=False)
    return coef_


# In[38]:


from keras.callbacks import LearningRateScheduler
def myScheduler(epoch):

    if epoch % 150 == 0 and epoch != 0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)
 
myReduce_lr = LearningRateScheduler(myScheduler)


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import random
import itertools
import logging
random.seed(555)  


def get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
    return list(nodes)

def get_nodes(net,num):
    net_nodes = [] 
    for i in range(1,num+1):
        net_nodes.append(get_nodes_at_level(net,i))    
    return net_nodes

def add_node(net,net_nodes):        
    for i in range(len(net_nodes)-2,-1,-1):
        data_temp = copy.deepcopy(net_nodes[i])
        for n in net_nodes[i]:
            nexts = net.successors(n)         
            temp = [ nex  for nex in nexts ] 
            if len(temp)==0:
                data_temp.remove(n)  # If the node of the current layer has no successor node, remove the node
            elif len(set(temp).intersection(set(net_nodes[i+1])))==0:   #if the subsequent node of the node of the current layer is not on the next layer, delete the node
                data_temp.remove(n)
            else:
                continue
        net_nodes[i] = data_temp
    return net_nodes


def get_note_relation(net_nodes):
    node_mat = []
    for i in range(len(net_nodes)-1):
        dicts = {}
        for n in net_nodes[i]:
            nexts = net.successors(n)  
            x = [ nex   for nex in nexts if nex in net_nodes[i+1] ]
            dicts[n] = x
        mat = np.zeros((len(net_nodes[i]), len(net_nodes[i+1]))) 
        for p, gs in dicts.items():     
            g_inds = [net_nodes[i+1].index(g) for g in gs]
            p_ind = net_nodes[i].index(p)
            mat[p_ind, g_inds] = 1
        df = pd.DataFrame(mat, index=net_nodes[i], columns=net_nodes[i+1])
        node_mat.append(df.T)
    return node_mat

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def add_edges(netx1,name):
    node1 = [node for node, indeg in netx1.in_degree() if indeg == 0]
    node2 = [node for node, indeg in netx1.out_degree() if indeg == 0]
    nodes = list(set(node1).intersection(set(node2)))
    edges = [(node,name) for node in nodes]
    return edges

def get_layers(network_genes,selected_genes,chosedPathways,netx1,netx2,netx3):
    layers = []
    nodes = network_genes
    dic = {}
    for n  in nodes:
        next = netx1.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = selected_genes
    for n  in nodes:
        next = netx2.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = chosedPathways
    for n  in nodes:
        next = netx3.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    return layers

def get_map_from_layer(layer_dict):
    genes = list(layer_dict.keys())
    #print(pathways)
    print ('genes', len(genes))
    pathways = list(itertools.chain.from_iterable(layer_dict.values()))
    pathways = list(np.unique(pathways))
    print ('pathways', len(pathways))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_genes, n_pathways))
    for g, ps in layer_dict.items():
        p_inds = [pathways.index(p) for p in ps]
        g_ind = genes.index(g)
        mat[g_ind, p_inds] = 1

    df = pd.DataFrame(mat, index=genes, columns=pathways)
    return df

def get_layer_maps(genes, layers,hs,num_omics,names):
    PK_layers = layers
    filtering_index = genes
    maps = []
    for i, layer in enumerate(PK_layers):
        print ('layer #', i)
        mapp = get_map_from_layer(layer)
        #print ('filtered_map', mapp.index)
        filter_df = pd.DataFrame(index=filtering_index)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        #print(filtered_map)
        #if i ==0:
        #    filtered_map = filtered_map[list(filtering_index)]
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print ('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        filtered_map = filtered_map.fillna(0)
        print ('filtered_map', filter_df.shape)
        if i==2:
            filtered_map = filtered_map[names]
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    names_hs0 = []
    for i in range(num_omics):
        num = 2 * hs[0]
        names = ['zhouli_'+str(i)+'_'+str(j) for j in range(num)]
        names_hs0.append(names)
        n1 = len(filtering_index)
        mat1 = np.concatenate([np.ones([n1,hs[0]]),np.ones([n1,hs[0]])],axis=1)
        df = pd.DataFrame(mat1, index=filtering_index, columns=names)
        maps.append(df)
    filtering_index = np.concatenate(names_hs0)
    for i in range(num_omics):
        mat1 = np.concatenate([np.ones([hs[0],hs[1]]),np.zeros([hs[0],hs[1]])],axis=1)
        mat2 = np.concatenate([np.zeros([hs[0],hs[1]]),np.ones([hs[0],hs[1]])],axis=1)
        mat3 = np.concatenate([mat1,mat2],axis=0)
        if i==0:
            mat = mat3
        else:
            mat = np.concatenate([mat,mat3],axis=0)
    num = 2 * hs[1]
    names = ['lijie_'+str(j) for j in range(num)]
    df = pd.DataFrame(mat, index=filtering_index, columns=names)
    maps.append(df)
    return maps

def get_layers_map_sec(subname,netx1,names):
    nodes = subname
    dic = {}
    for n  in nodes:
        next = netx1.successors(n)
        dic[n] = [nex for nex in next]
    filtering_index = subname
    mapp = get_map_from_layer(dic)
    filter_df = pd.DataFrame(index=filtering_index)
    filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
    filtered_map = filtered_map.fillna(0)
    filtered_map = filtered_map[names]
    return filtered_map

def load_data_dict(filename):
    data_dict_list = []
    dict = {}
    with open( filename) as gmt:
        data_list = gmt.readlines()
        # print data_list[0]
        for row in data_list:
            genes = row.split('\t')
            genes = [ i.replace('\n','') for i in genes]
            dict[genes[2]] = genes[3:]
    return dict
# # pathways process and network generation



# In[6]:
kfscore = []
kfscore1 = []
#cancer = 'BRCAB'
#cancer1 = 'BRCA'
cancer = 'PPRAD'  #('PBRCA','BRCA')
cancer1 = 'PRAD'
num_runs = 5
threa = 0.25
threa1 = 0.5
n_hidden_layers_PtH = [24,24] 
omics_num = 4
n_knn = 11
h_names = ['nor','tum','com']


# loading mrna data
file = "./data/cc/" + cancer + '_mut_matrix.csv'
#mrna_data = pd.read_csv("./data/TCGA-LUAD.varscan2_snv.csv",index_col = 0)
mrna_data = pd.read_csv(file,index_col = 0)
#loading cnv data
file = "./data/cc/" + cancer + '_cnv_matrix.csv'
cnv_data = pd.read_csv(file,index_col = 0)
#cnv_data = pd.read_csv("./data/TCGA-LUAD_cnv.csv",index_col = 0)
#loading cnv data
#loading label
file = "./data/cc/response_paper_" + cancer + '.csv'
response  = pd.read_csv(file,index_col=0)
#response  = pd.read_csv('./data/response_paper.csv',index_col=0)


#read network genes
network_genes = pd.read_csv('./data/genes/HN_genes.csv', sep='\t',dtype=str)
#network_genes = pd.read_csv('./data/genes/HN_genes_0.1.csv', sep='\t',dtype=str)

# Read gene-gene relationships 
network_edges = pd.read_csv('./data/networks/HumanNet.txt', sep='\t',names=['start','end'],dtype=str)
#network_edges = pd.read_csv('./data/networks/HumanNet_0.1.txt', sep='\t',names=['start','end'],dtype=str)
network_edges_copy = copy.deepcopy(network_edges)
network_edges_copy.columns = ['end','start']
network_edges_copy = network_edges_copy[['start','end']]
network_edges = pd.concat([network_edges, network_edges_copy], ignore_index=True)
network_edges = network_edges.drop_duplicates()


#Disrupted data set
response = response.sample(frac=1)
mrna_data = mrna_data.sample(frac=1)
cnv_data = cnv_data.sample(frac=1)


#split copy number variation data
import copy
cnv_amp = copy.deepcopy(cnv_data)

#cnv_data
cnv_amp[cnv_amp <= 0] = 0.
cnv_amp[cnv_amp > 0 ] = 1.

#dmeth_data
cnv_data[cnv_data >= 0] = 0.
cnv_data[cnv_data < 0 ] = 1.
cnv_del = cnv_data

# In[16]:
print(np.array(cnv_amp.values.nonzero()).shape)
print(np.array(cnv_del.values.nonzero()).shape)
print(np.array(mrna_data.values.nonzero()).shape)

#
#print(cnv_amp)
mrna_data = mrna_data.loc[response.index]
cnv_amp = cnv_amp.loc[response.index]
cnv_del = cnv_del.loc[response.index]
mrna_data_genes = mrna_data.columns
cnv_amp_genes = cnv_amp.columns
cnv_del_genes = cnv_del.columns
mrna_data_genenum = len(mrna_data_genes)
cnv_amp_genenum = len(cnv_amp_genes)
cnv_del_genenum = len(cnv_del_genes)


re_cnv_amp = copy.deepcopy(cnv_amp)
re_cnv_del = copy.deepcopy(cnv_del)
re_mrna_data = copy.deepcopy(mrna_data)
re_response = copy.deepcopy(response)


# Read gene-pathway annotation relationships
pathway_genes = pd.read_csv('./data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
pathways = list(set(pathway_genes['group']))
pathway_num = len(pathways)
pathway_genes_num = {}
pathway_mrna = {}
pathway_amp = {}
pathway_del = {}
pathway_dmeth = {}
for h in range(pathway_num):
    pathway = pathways[h]
    aa = []
    aa.append(pathway)
    pathway_gene = pathway_genes[pathway_genes['group'].isin(aa)]
    genes = pathway_gene['gene']
    genes_inter_mrna = list(set(genes).intersection(set(mrna_data_genes)))
    genes_inter_amp = list(set(genes).intersection(set(cnv_amp_genes)))
    genes_inter_del = list(set(genes).intersection(set(cnv_del_genes)))
    record_num = []
    record_num.append(len(genes_inter_mrna))
    record_num.append(len(genes_inter_amp))
    record_num.append(len(genes_inter_del))
    pathway_genes_num[pathway] = record_num
    pathway_mrna[pathway] = genes_inter_mrna
    pathway_amp[pathway] = genes_inter_amp
    pathway_del[pathway] = genes_inter_del


x11 = mrna_data.values
y1 = response['response'].values
y1 = y1.reshape(-1)
chosefea = 5000
istrain = False
data_type_list =['mrna_data','cnv_amp','cnv_del']
Save_Res = []
Save_Res1 = []
Save_Res2 = []
Save_Res3 = []
Save_Res4 = []
for rs in range(4,num_runs):
    kfscore = []
    kfscore1 = []
    kfscore2 = []
    kfscore3 = []
    kfscore4 = []
    #p = 0
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=rs+1)
    #skf = RepeatedKFold(n_splits=5,n_repeats=2,shuffle=True,random_state=rs+1)
    for train_index, test_index in skf.split(x11, y1):
        #***
        cnv_amp = copy.deepcopy(re_cnv_amp)
        cnv_del = copy.deepcopy(re_cnv_del)
        mrna_data = copy.deepcopy(re_mrna_data)
        response = copy.deepcopy(re_response)
        if istrain:
            skf1 = StratifiedKFold(n_splits=5,shuffle=True,random_state=rs+2)
            y12 = y1[train_index]
            for train_index1, test_index1 in skf1.split(train_index, y12):
                break
            tra1 = train_index[train_index1]
            tes1 = train_index[test_index1]
            response1 =  response.iloc[tra1,:]
            response2 =  response.iloc[tes1,:]
        else:
            response1 =  response.iloc[train_index,:]
            response2 =  response.iloc[test_index,:]
        tol_snv = mrna_data.join(response1,how='inner')
        model = SelectKBest(chi2, k=chosefea)
        #print(tol_snv.values[:,-1])
        temp = tol_snv.values[:,0:-1]
        temp = np.nan_to_num(temp, nan=0.0)
        x_data1 = model.fit_transform(temp,tol_snv.values[:,-1])
        #x_data1 = model.fit_transform(tol_snv.values[:,4300:4632],tol_snv.values[:,-1])
        fea = model.get_support()
        mrna_data = mrna_data.loc[:,fea]
        mrna_data_chosegenes = list(mrna_data.columns)

        #cnv_amp
        tol_amp = cnv_amp.join(response1,how='inner')
        tol_amp = tol_amp.fillna(0)
        model = SelectKBest(chi2, k=chosefea)
        x_data1 = model.fit_transform(tol_amp.values[:,0:-1],tol_amp.values[:,-1])
        fea = model.get_support()
        cnv_amp = cnv_amp.loc[:,fea]
        cnv_amp_chosegenes = list(cnv_amp.columns)

        #cnv_del
        tol_amp = cnv_del.join(response1,how='inner')
        tol_amp = tol_amp.fillna(0)
        model = SelectKBest(chi2, k=chosefea)
        x_data1 = model.fit_transform(tol_amp.values[:,0:-1],tol_amp.values[:,-1])
        fea = model.get_support()
        cnv_del = cnv_del.loc[:,fea]
        cnv_del_chosegenes = list(cnv_del.columns)
        
        mrna_pvalues = []
        amp_pvalues = []
        del_pvalues = []
        for h in range(pathway_num):
            pathway = pathways[h]
            genes_inter_mrna = pathway_mrna[pathway]
            record_num = pathway_genes_num[pathway]
            pathway_inter_gene =  set(mrna_data_chosegenes).intersection(set(genes_inter_mrna))
            a1 =record_num[0]
            a2 = len(pathway_inter_gene)
            a3 = a1 - a2
            a4 = chosefea - a2
            a5 = mrna_data_genenum - a1 - a4
            table = np.array([[a2, a3],[a4, a5]])
            odds_ratio, p_value = ss.fisher_exact(table,alternative='greater')
            mrna_pvalues.append(p_value)
            #
            genes_inter_amp = pathway_amp[pathway]
            pathway_inter_gene =  set(cnv_amp_chosegenes).intersection(set(genes_inter_amp))
            a1 =record_num[1]
            a2 = len(pathway_inter_gene)
            a3 = a1 - a2
            a4 = chosefea - a2
            a5 = cnv_amp_genenum - a1 - a4
            table = np.array([[a2, a3],[a4, a5]])
            odds_ratio, p_value = ss.fisher_exact(table,alternative='greater')
            amp_pvalues.append(p_value)
            #
            genes_inter_del = pathway_del[pathway]
            pathway_inter_gene =  set(cnv_del_chosegenes).intersection(set(genes_inter_del))
            a1 =record_num[2]
            a2 = len(pathway_inter_gene)
            a3 = a1 - a2
            a4 = chosefea - a2
            a5 = cnv_del_genenum - a1 - a4 
            table = np.array([[a2, a3],[a4, a5]])
            odds_ratio, p_value = ss.fisher_exact(table,alternative='greater')
            del_pvalues.append(p_value)
                       
        pvalues = pd.DataFrame([mrna_pvalues, amp_pvalues,del_pvalues]).transpose()
        pvalues.index = pathways
        pvalues.columns = data_type_list
        temp1 = pvalues[pvalues['mrna_data']<=threa]
        mrna_pathways = list(temp1.index)
        print(len(mrna_pathways))
        temp1 = pvalues[pvalues['cnv_amp']<=threa]
        amp_pathways = list(temp1.index)
        print(len(amp_pathways))
        temp1 = pvalues[pvalues['cnv_del']<=threa]
        del_pathways = list(temp1.index)
        print(len(del_pathways))
        


        # In[18]:
        print(mrna_data.shape)
        print(cnv_amp.shape)
        print(cnv_del.shape)

        x_list = []
        y_list = []
        rows_list = []
        cols_list = []

        #data_type_list =['mrna_data']

        for ind in [mrna_data,cnv_amp,cnv_del]: 
            get_data = ind.join(response,how='inner')
            del get_data['response']
            
            row = get_data.index      
            col = get_data.columns     
            resp = response.loc[row]   
            
            x_list.append(ind)
            y_list.append(resp)
            rows_list.append(row)
            cols_list.append(col)

        x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type = 'union')
        union_gene = list(cols.levels[0])

        #Omics_data = x[list(gene_pathway_df.columns)]
        Omics_data = x[union_gene]        
        Omics_data.shape
        # In[28]:
        cols = union_gene  
        #mapp = Get_Node_relation[0].values         
        # In[29]:
        #mapp
        x = Omics_data.values
        y = y.reshape(-1)
        x_0 =  0.61
        x_1 =  2.61
        #GBM(0.61,2.61);LGG(1.54,0.74);KIRC(1.49,0.715)
        ind_train = rows.isin(response1.index)
        ind_test = rows.isin(response2.index)
        X_train = x[ind_train]
        y_train = y[ind_train]
        X_test = x[ind_test]
        y_test = y[ind_test]
        #snv_data; cnv_amp; cnv_del
        single_mrna =Omics_data.swaplevel(i=0, j=1, axis=1)['mrna_data']
        #single_mrna = imputation(single_mrna,barcodes,min_dist_indices)
        x1 = single_mrna.values
        X_train1 = x1[ind_train]
        X_test1 = x1[ind_test]
        data_mrna = single_mrna.iloc[ind_train,:]

        single_cnv_amp =Omics_data.swaplevel(i=0, j=1, axis=1)['cnv_amp']
        #single_cnv_amp = imputation(single_cnv_amp,barcodes,min_dist_indices)
        x1 = single_cnv_amp.values
        X_train2 = x1[ind_train]
        X_test2 = x1[ind_test]
        data_amp = single_cnv_amp.iloc[ind_train,:]

        single_cnv_del =Omics_data.swaplevel(i=0, j=1, axis=1)['cnv_del']
        #single_cnv_del = imputation(single_cnv_del,barcodes,min_dist_indices)
        x1 = single_cnv_del.values
        X_train3 = x1[ind_train]
        X_test3 = x1[ind_test]
        data_del = single_cnv_del.iloc[ind_train,:]
        
        #
        network_edges_new = copy.deepcopy(network_edges)
        union_network_gene =list(set(union_gene).intersection(set(network_genes['genes'])))
        if len(union_network_gene)!=len(union_gene):
            nodes = set(union_gene).difference(union_network_gene)
            df = {}
            df['start'] = list(nodes)
            df['end'] = list(nodes)
            df = pd.DataFrame(df)
            network_edges_new = pd.concat([network_edges_new, df], ignore_index=True)

        # mrna_data:
        network_edges_mrna = network_edges_new[(network_edges_new['start'].isin(union_gene)) & (network_edges_new['end'].isin(mrna_data_chosegenes))]
        net1 = nx.from_pandas_edgelist(network_edges_mrna, 'start', 'end', create_using=nx.DiGraph())
        net1.name = 'HumanNet'
        nodes = set(union_gene).difference(set(net1.nodes))
        if len(nodes)>0:
            root_node = 'mrna_0'
            edges = [(node,root_node) for node in nodes]
            net1.add_edges_from(edges)
            mrna_data_chosegenes.append('mrna_0')
        # Read gene-pathway annotation relationships
        #pathway_genes = pd.read_csv('./data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
        pathway_genes_mrna = pathway_genes[(pathway_genes['gene'].isin(mrna_data_chosegenes)) & (pathway_genes['group'].isin(mrna_pathways))]
        net2 = nx.from_pandas_edgelist(pathway_genes_mrna, 'gene', 'group', create_using=nx.DiGraph())
        net2.name = 'GeneToPathway_mrna'
        nodes = set(mrna_data_chosegenes).difference(set(pathway_genes_mrna['gene']))
        if len(nodes)>0:
            root_node = 'mrna_1'
            edges = [(node,root_node) for node in nodes]
            net2.add_edges_from(edges)
            mrna_pathways.append(root_node)
        #
        mgsea_result_mrna = cal_mgsea(response1,cancer,'mrna',mrna_pathways)
        temp_result = mgsea_result_mrna[mgsea_result_mrna['nor_pvalue']<=threa1]
        mrna_nor_pathways = list(temp_result.index)
        temp_result = mgsea_result_mrna[mgsea_result_mrna['tum_pvalue']<=threa1]
        mrna_tum_pathways = list(temp_result.index)
        net3,mrna_names = createNetwork1(h_names,'mrna_data',mrna_pathways,mrna_nor_pathways,mrna_tum_pathways,n_hidden_layers_PtH[0])
        #
        layers = get_layers(union_gene,mrna_data_chosegenes,mrna_pathways,net1,net2,net3)
        Get_Node_relation_mrna = get_layer_maps(union_gene, layers,n_hidden_layers_PtH,omics_num,mrna_names)

        #snv_amp
        # In[24]:
        network_edges_amp = network_edges_new[(network_edges_new['start'].isin(union_gene)) & (network_edges_new['end'].isin(cnv_amp_chosegenes))]
        net1 = nx.from_pandas_edgelist(network_edges_amp, 'start', 'end', create_using=nx.DiGraph())
        net1.name = 'HumanNet'
        nodes = set(union_gene).difference(set(net1.nodes))
        if len(nodes)>0:
            root_node = 'amp_0'
            edges = [(node,root_node) for node in nodes]
            net1.add_edges_from(edges)
            cnv_amp_chosegenes.append('amp_0')
        # Read gene-pathway annotation relationships
        #pathway_genes = pd.read_csv('./data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
        pathway_genes_amp = pathway_genes[(pathway_genes['gene'].isin(cnv_amp_chosegenes)) & (pathway_genes['group'].isin(amp_pathways))]
        net2 = nx.from_pandas_edgelist(pathway_genes_amp, 'gene', 'group', create_using=nx.DiGraph())
        net2.name = 'GeneToPathway_mrna'
        nodes = set(cnv_amp_chosegenes).difference(set(pathway_genes_amp['gene']))
        if len(nodes)>0:
            root_node = 'amp_1'
            edges = [(node,root_node) for node in nodes]
            net2.add_edges_from(edges)
            amp_pathways.append(root_node)
        #
        mgsea_result_amp = cal_mgsea(response1,cancer,'cnv_amp',amp_pathways)
        temp_result = mgsea_result_amp[mgsea_result_amp['nor_pvalue']<=threa1]
        amp_nor_pathways = list(temp_result.index)
        temp_result = mgsea_result_amp[mgsea_result_amp['tum_pvalue']<=threa1]
        amp_tum_pathways = list(temp_result.index)
        net3,amp_names = createNetwork1(h_names,'amp_data',amp_pathways,amp_nor_pathways,amp_tum_pathways,n_hidden_layers_PtH[0]) 
        #
        layers = get_layers(union_gene,cnv_amp_chosegenes,amp_pathways,net1,net2,net3)
        Get_Node_relation_amp = get_layer_maps(union_gene, layers,n_hidden_layers_PtH,omics_num,amp_names)

        #snv_del
        # In[24]:
        network_edges_del = network_edges_new[(network_edges_new['start'].isin(union_gene)) & (network_edges_new['end'].isin(cnv_del_chosegenes))]
        net1 = nx.from_pandas_edgelist(network_edges_del, 'start', 'end', create_using=nx.DiGraph())
        net1.name = 'HumanNet'
        nodes = set(union_gene).difference(set(net1.nodes))
        if len(nodes)>0:
            root_node = 'del_0'
            edges = [(node,root_node) for node in nodes]
            net1.add_edges_from(edges)
            cnv_del_chosegenes.append('del_0')
        # Read gene-pathway annotation relationships
        #pathway_genes = pd.read_csv('./data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
        pathway_genes_del = pathway_genes[(pathway_genes['gene'].isin(cnv_del_chosegenes)) & (pathway_genes['group'].isin(del_pathways))]
        net2 = nx.from_pandas_edgelist(pathway_genes_del, 'gene', 'group', create_using=nx.DiGraph())
        net2.name = 'GeneToPathway_mrna'
        nodes = set(cnv_del_chosegenes).difference(set(pathway_genes_del['gene']))
        if len(nodes)>0:
            root_node = 'del_1'
            edges = [(node,root_node) for node in nodes]
            net2.add_edges_from(edges)
            del_pathways.append(root_node)
        # 
        mgsea_result_del = cal_mgsea(response1,cancer,'cnv_del',del_pathways)
        temp_result = mgsea_result_del[mgsea_result_del['nor_pvalue']<=threa1]
        del_nor_pathways = list(temp_result.index)
        temp_result = mgsea_result_del[mgsea_result_del['tum_pvalue']<=threa1]
        del_tum_pathways = list(temp_result.index)
        net3,del_names = createNetwork1(h_names,'del_data',del_pathways,del_nor_pathways,del_tum_pathways,n_hidden_layers_PtH[0]) 
        layers = get_layers(union_gene,cnv_del_chosegenes,del_pathways,net1,net2,net3)
        Get_Node_relation_del = get_layer_maps(union_gene, layers,n_hidden_layers_PtH,omics_num,del_names)
        
        net4,subnames,allnames = createNetwork4(mrna_names,amp_names,del_names,n_hidden_layers_PtH) 
        Get_Node_relation_next = get_layers_map_sec(subnames,net4,allnames)
        net4,subnames,allnames = createNetwork5(mrna_names,amp_names,n_hidden_layers_PtH) 
        Get_Node_relation_next1 = get_layers_map_sec(subnames,net4,allnames)
        net4,subnames,allnames = createNetwork5(mrna_names,del_names,n_hidden_layers_PtH) 
        Get_Node_relation_next2 = get_layers_map_sec(subnames,net4,allnames)
        net4,subnames,allnames = createNetwork5(amp_names,del_names,n_hidden_layers_PtH) 
        Get_Node_relation_next3 = get_layers_map_sec(subnames,net4,allnames)

        #model = create_models(x1)
        model = create_models_pheno11(x1)

        #history = model.fit([X_train1,X_train2,X_train3,X_train4,age_train],[y_train]*5,epochs=150,class_weight={0:x_0,1:x_1},batch_size = 20)  #epochs=50,class_weight={0:0.5,1:1.35},batch_size = 64
        #history = model.fit(X_train,y_train,epochs=120,class_weight="balanced",batch_size = 64)  #epochs=50,class_weight={0:0.5,1:1.35},batch_size = 64
        history = model.fit([X_train1,X_train2,X_train3],[y_train]*7,epochs=75,batch_size = 32,shuffle=True)
        #history = model.fit([X_train1,X_train2,X_train3,X_train4,age_train],[y_train]*5,epochs=80,batch_size = 30,shuffle=True)

        y_pred = model.predict([X_test1,X_test2,X_test3])
        kfscore.append(evaluates(y_test, y_pred[-1]))
        results = evaluates(y_test, y_pred[-1])
        print("results : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(results[0],3),round(results[1],3),round(results[2],3),round(results[3],3),round(results[4],3),round(results[5],3),round(results[6],3)))
        y_pred1 = np.mean(np.array(y_pred[4:7]),axis=0)
        kfscore1.append(evaluates(y_test, y_pred1))
        results = evaluates(y_test, y_pred1)
        print("results : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(results[0],3),round(results[1],3),round(results[2],3),round(results[3],3),round(results[4],3),round(results[5],3),round(results[6],3)))
        y_pred1 = np.mean(np.array(y_pred)[[3,4,6]],axis=0)
        kfscore2.append(evaluates(y_test, y_pred1))
        results = evaluates(y_test, y_pred1)
        print("results : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(results[0],3),round(results[1],3),round(results[2],3),round(results[3],3),round(results[4],3),round(results[5],3),round(results[6],3)))
        y_pred1 = np.mean(np.array(y_pred),axis=0)
        kfscore3.append(evaluates(y_test, y_pred1))
        results = evaluates(y_test, y_pred1)
        print("results : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(results[0],3),round(results[1],3),round(results[2],3),round(results[3],3),round(results[4],3),round(results[5],3),round(results[6],3)))
        y_pred1 = np.mean(np.array(y_pred)[[0,1,3,4,6]],axis=0)
        kfscore4.append(evaluates(y_test, y_pred1))
        results = evaluates(y_test, y_pred1)
        print("results : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(results[0],3),round(results[1],3),round(results[2],3),round(results[3],3),round(results[4],3),round(results[5],3),round(results[6],3)))
        # feature importance
        #explain_x = X_train[np.where(y_train!=0)]
        #explain_y = y_train[np.where(y_train!=0)]
        #coef_ = get_coef_importance(model,explain_x, explain_y, target=-1,feature_importance='deepexplain_deeplift')
        #cof_values = ['inputs','h_0','h0','h1','h2','h3','h4']
        #name = [np.array(Omics_data.columns),gene_pathway_df.columns,Get_Node_relation[3].index,Get_Node_relation[2].index,Get_Node_relation[1].index,Get_Node_relation[0].index,Get_Node_relation[0].columns]
    #     os.mkdir('./data/coef/h{}/'.format(p))
        #for i in range(0,7):
        #    X = pd.DataFrame()
        #    X['name'] = name[i]
        #    X['values'] = coef_[0][cof_values[i]]
        #    X.to_csv('./data/coef/h{}/{}.csv'.format(p,cof_values[i]),index=False,encoding='UTF-8')
        #p = p + 1
        del model
    #avrrage
    kfscores = np.array(kfscore).sum(axis= 0)/5.0
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(results[6],3)))
    Save_Res.append(kfscores)
    resu = pd.DataFrame(kfscore)
    resu.to_csv('./data/result/{}/{}_{}_{}'.format(cancer,threa,threa1,0),index = False)
    kfscores = np.array(kfscore1).sum(axis= 0)/5.0
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(results[6],3)))
    Save_Res1.append(kfscores)
    resu = pd.DataFrame(kfscore1)
    resu.to_csv('./data/result/{}/{}_{}_{}'.format(cancer,threa,threa1,1),index = False)
    kfscores = np.array(kfscore2).sum(axis= 0)/5.0
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(results[6],3)))
    Save_Res2.append(kfscores)
    resu = pd.DataFrame(kfscore2)
    resu.to_csv('./data/result/{}/{}_{}_{}'.format(cancer,threa,threa1,2),index = False)
    kfscores = np.array(kfscore3).sum(axis= 0)/5.0
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(results[6],3)))
    Save_Res.append(kfscores)
    resu = pd.DataFrame(kfscore3)
    resu.to_csv('./data/result/{}/{}_{}_{}'.format(cancer,threa,threa1,3),index = False)
    kfscores = np.array(kfscore4).sum(axis= 0)/5.0
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(results[6],3)))
    Save_Res.append(kfscores)
    resu = pd.DataFrame(kfscore4)
    resu.to_csv('./data/result/{}/{}_{}_{}'.format(cancer,threa,threa1,4),index = False)


file = "./data/result/" + cancer + '/avg_results.csv'  
np.savetxt(file, np.array(Save_Res),delimiter='\t')  
