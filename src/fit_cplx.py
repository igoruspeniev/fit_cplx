import numpy as np
import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import math

np.set_printoptions(threshold=np.nan)
# just to know correct answer for y = sin(pi*x + 1):
# C1=0.42073549-0.27015115*i, L1=0+PI*i, C2=0.42073549+0.27015115*i, L2=0-PI*i


def f(x): return math.sin(math.pi * x + 1)


def init_weight(): return np.random.uniform(-5, 5)


def randomise_vars(variables, sess: tf.Session):
    for variable in variables:
        op = variable.assign(init_weight())
        sess.run(op)


x_from = -5  # начало интервала
x_to = 5  # конец интервала
noise_STD = 0.1  # среднеквадратическое отклонение шума
interval_count = 1024  # количество интервалов
packet_size = interval_count  # размер пакета

np.random.seed(0)  # делаем случайность предсказуемой чтобы повторять вычисления на этом же наборе данных
data_x = np.arange(x_from, x_to, (x_to - x_from) / interval_count)
np.random.shuffle(data_x)  # перемешиваем
data_y = list(map(f, data_x)) + np.random.normal(0, noise_STD, interval_count)
print(",".join(list(map(str, data_x[:packet_size]))))  # первый пакет аргументов
print(",".join(list(map(str, data_y[:packet_size]))))  # первый пакет значений

tf_data_x = tf.placeholder(dtype=tf.float64, shape=[packet_size])  # узел на который подаются аргументы функции
tf_data_y = tf.placeholder(dtype=tf.float64, shape=[packet_size])  # узел на который подаются значения функции

Cr1 = tf.Variable(initial_value=0, dtype=tf.float64, name='Cr1')
Ci1 = tf.Variable(initial_value=0, dtype=tf.float64, name='Ci1')
Lr1 = tf.Variable(initial_value=0, dtype=tf.float64, name='Lr1')
Li1 = tf.Variable(initial_value=0, dtype=tf.float64, name='Li1')
Cr2 = tf.Variable(initial_value=0, dtype=tf.float64, name='Cr2')
Ci2 = tf.Variable(initial_value=0, dtype=tf.float64, name='Ci2')
Lr2 = tf.Variable(initial_value=0, dtype=tf.float64, name='Lr2')
Li2 = tf.Variable(initial_value=0, dtype=tf.float64, name='Li2')
e1 = tf.multiply(tf.complex(Cr1, Ci1),
                 tf.exp(tf.multiply(tf.complex(Lr1, Li1), tf.cast(tf_data_x, dtype=tf.complex128))))
e2 = tf.multiply(tf.complex(Cr2, Ci2),
                 tf.exp(tf.multiply(tf.complex(Lr2, Li2), tf.cast(tf_data_x, dtype=tf.complex128))))
model_c = tf.real(tf.add(e1, e2))

best_invloss = 0
bcr1 = 0
bci1 = 0
bcr2 = 0
bci2 = 0
blr1 = 0
bli1 = 0
blr2 = 0
bli2 = 0

if __name__ == "__main__":
    loss = tf.reduce_mean(tf.square(model_c - tf_data_y))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999).minimize(loss)
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # session.run(tf.variables_initializer([Lr1, Lr2, Li1, Li2, Ci1, Ci2, Cr1, Cr2]))

        for epoch in range(1):
            print("=== epoch #%d ===" % epoch)
            np.random.seed(epoch)
            randomise_vars([Cr1, Ci1, Lr1, Li1, Cr2, Ci2, Lr2, Li2], session)
            print('C1 = (%0.4f, %0.4fi), L1 = (%0.4f, %0.4fi), C2 = (%0.4f, %0.4fi), L2 = (%0.4f, %0.4fi)'
                  % (Cr1.eval(), Ci1.eval(), Lr1.eval(), Li1.eval(), Cr2.eval(), Ci2.eval(), Lr2.eval(), Li2.eval()))

            for i in range(interval_count // packet_size):
                feed_dict = {tf_data_x: data_x[i * packet_size:(i + 1) * packet_size],
                             tf_data_y: data_y[i * packet_size:(i + 1) * packet_size]}
                # print(feed_dict)
                for j in range(10000):
                    _, loss_value = session.run([optimizer, loss], feed_dict=feed_dict)
                    print('%d: C1 = (%0.4f, %0.4fi), L1 = (%0.4f, %0.4fi), C2 = (%0.4f, %0.4fi), L2 = (%0.4f, %0.4fi), loss = %f'
                      % (j, Cr1.eval(), Ci1.eval(), Lr1.eval(), Li1.eval(), Cr2.eval(), Ci2.eval(), Lr2.eval(), Li2.eval(), loss_value))
                invloss = 1 / (1 + loss_value)
                if invloss > best_invloss:
                    best_invloss = invloss
                    bcr1 = Cr1.eval()
                    bci1 = Ci1.eval()
                    blr1 = Lr1.eval()
                    bli1 = Li1.eval()
                    bcr2 = Cr2.eval()
                    bci2 = Ci2.eval()
                    blr2 = Lr2.eval()
                    bli2 = Li2.eval()

    print(bcr1, bci1, "C1")
    print(bcr2, bci2, "C2")
    print(blr1, bli1, "L1")
    print(blr2, bli2, "L2")
    c1 = np.complex(bcr1, bci1)
    c2 = np.complex(bcr2, bci2)
    l1 = np.complex(blr1, bli1)
    l2 = np.complex(blr2, bli2)
    plt.plot(data_x, data_y, 'ro')
    data_x_sorted = np.arange(x_from, x_to, (x_to - x_from) / 1000)
    plt.plot(data_x_sorted, list(map(lambda x: (c1 * np.exp(l1 * x) + c2 * np.exp(l2 * x)), data_x_sorted)))
    plt.show()
