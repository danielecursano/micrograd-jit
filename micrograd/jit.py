from micrograd.engine import Value
import subprocess
import ctypes
import tempfile
import os
import numpy as np

class InputNode(Value):
    ref = 0

    def __init__(self, data=float("inf")):
        super().__init__(data)
        self.id = InputNode.ref
        InputNode.ref += 1

    def __hash__(self):
        return self.id

def repr(node: Value):
    return f"v_{hash(node)}"

def trace(root: Value):
    seen = set()
    nodes = []

    def build(v: Value):
        if v in seen:
            return
        seen.add(v)
        for child in v._prev:
            build(child)
        nodes.append(v)

    build(root)

    inputs = []
    graph = []

    for node in nodes:
        node_name = repr(node)
        if len(node._prev) == 0:
            if isinstance(node, InputNode):
                inputs.append(node_name)
            else:
                graph.append((node_name, [node.data], "def"))
        else:
            graph.append((node_name, [repr(c) for c in node._prev], node._op))

    return graph, inputs


def generate_c_code(graph, inputs):
    code = f"float f({', '.join(f'float {inp}' for inp in inputs)}) " + "{\n"
    for dest, src, op in graph:
        if op == "def":
            code += f"\tconst float {dest} = {src[0]};\n"
        elif op == "ReLU":
            code += f"\tfloat {dest} = {src[0]} > 0 ? {src[0]} : 0;\n"
        else:
            code += f"\tfloat {dest} = {src[0]} {op} {src[1]};\n"
    code += f"\treturn {graph[-1][0]};\n"
    code += "}"
    return code

def parse_input(func):
    return [InputNode() for _ in range(func.__code__.co_argcount)]

class JitFunction:
    def __init__(self, f, n_args):
        self.n_args = n_args
        self.f = f

    def __call__(self, *inp):
        # If a single argument is passed and it's a 1D array or list, treat it as a row
        if len(inp) == 1:
            X = np.asarray(inp[0], dtype=np.float32)
            if X.ndim == 1:
                # 1D array/list of length n_args → expand to shape (1, n_args)
                if len(X) == self.n_args:
                    inp = X[np.newaxis, :]
                else:
                    raise ValueError(f"Expected {self.n_args} arguments, got {len(X)}")
            else:
                # Already 2D, use as is
                inp = X
        # Map row by row
        return np.array([self.f(*row) for row in inp])

def jit(func):
    inputs = parse_input(func)
    output = func(*inputs)
    graph, inputs = trace(output)
    inputs.sort(key=lambda x: x.split("_")[1])
    c_code = generate_c_code(graph, inputs)   
    with tempfile.TemporaryDirectory() as d:
        c_path = os.path.join(d, "jit.c")
        so_path = os.path.join(d, "jit.so")

        with open(c_path, "w") as f:
            f.write(c_code)

        subprocess.check_call([
            "gcc", "-O3", "-shared", "-fPIC", c_path, "-o", so_path
        ])

        lib = ctypes.CDLL(so_path)
        lib.f.argtypes = [ctypes.c_float, ctypes.c_float]
        lib.f.restype = ctypes.c_float

        return JitFunction(lib.f, len(inputs))
