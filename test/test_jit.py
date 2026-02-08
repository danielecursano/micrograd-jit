from micrograd.nn import Neuron, Layer, MLP
from micrograd.engine import Value
from micrograd.jit import jit
import numpy as np

def test_function():

    def f(x, y):
        return (x * y + 2).relu()
    
    jit_f = jit(f)

    assert np.isclose(jit_f([1, 2]), f(Value(1), Value(2)).data)

def test_more_than_2_args():

    def f(x, y, z, w):
        return (x * y + 2).relu() * (z+w)
    
    jit_f = jit(f)
    assert jit_f.n_args == 4
    assert np.isclose(jit_f([1, 2, 3, 4]), f(Value(1), Value(2), Value(3), Value(4)).data)

def test_mlp():

    model = MLP(2, [16, 16, 1]) 

    def forward(x, y):
        return model([x, y])
    
    jit_f = jit(forward)

    X = np.random.normal(size=(100, 2))
    output = np.array([x.data for x in list(map(model, X))], dtype=np.float32)
    total_diff = np.sum(np.abs(list(map(jit_f, X)) - output))
    assert total_diff < 1e-4


if __name__ == "__main__":
    test_function()
    test_more_than_2_args
    test_mlp()
    print("All jit tests passed!")