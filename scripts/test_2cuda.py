import jax
import jax.numpy as jnp
from jax import jit, device_put

# 定义一个简单的矩阵乘法函数
@jit
def matmul(a, b):
    return jnp.dot(a, b)

def main():
    # 设定环境变量，指定使用的显卡编号，这里假设只使用显卡编号为 0
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 创建两个随机矩阵，放到设备上进行计算
    a = jax.random.normal(jax.random.PRNGKey(0), (2000, 2000))
    b = jax.random.normal(jax.random.PRNGKey(1), (2000, 2000))
    a_gpu = device_put(a)
    b_gpu = device_put(b)

    # 执行矩阵乘法
    result = matmul(a_gpu, b_gpu)

    # 打印结果的部分信息
    print("Result shape:", result.shape)
    print("Result example:", result[:5, :5])

if __name__ == "__main__":

    main()

