from compiler import compile_python_to_dag
import networkx as nx

with open("kernel.py", "r") as f:
    source = f.read()

dag = compile_python_to_dag(source)
print("Nodes in DAG:", len(dag.nodes()))
for n, data in dag.nodes(data=True):
    print(f"Node {n}: {data}")

print("Edges:", dag.edges())
