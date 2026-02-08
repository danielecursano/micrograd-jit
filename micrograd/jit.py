from micrograd.engine import Value
import subprocess
import ctypes
import tempfile
import os
import numpy as np

class InputNode(Value):
    """
    A Value subclass used to build an IR for JIT compilation.

    Each InputNode is assigned a unique integer ID, which is used
    for hashing and for generating a stable, sorted input argument
    list for the generated C function.
    """
    ref = 0

    def __init__(self, data=float("inf")):
        super().__init__(data)
        self.id = InputNode.ref
        InputNode.ref += 1

    def __hash__(self):
        return self.id

def repr(node: Value):
    """
    Return a stable symbolic name for a Value node.

    :param node: Value node to be named.
    :type node: Value
    :return: Symbolic name of the node.
    :rtype: str
    """
    return f"v_{hash(node)}"

def trace(root: Value):
    """
    Trace a computation graph from the output node back to its inputs
    and build an intermediate representation (IR).

    Returns a topologically sorted graph and the corresponding input nodes.
    """
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
    """
    Generate the C code given a computation graph and the list of arguments of the function.
    """
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
    """
    Generate InputNode placeholders for each argument of the function
    to be JIT-compiled.
    """
    return [InputNode() for _ in range(func.__code__.co_argcount)]

class JitFunction:
    """
    A callable wrapper for a JIT-compiled function.

    This class stores a compiled function `f` and the expected number
    of arguments. Calling an instance checks that the input length
    matches `n_args` and then calls the underlying function. 
    """
    def __init__(self, f, n_args):
        self.n_args = n_args
        self.f = f

    def __call__(self, inp):
        assert len(inp) == self.n_args
        return self.f(*inp)

def jit(func):
    """
    JIT-compile a Python function into a C implementation.
    """
    # TODO: Finalize numeric types across Python ↔ C JIT boundary.
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
