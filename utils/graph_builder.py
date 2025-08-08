# utils/graph_builder.py

import networkx as nx

def build_gate_interaction_graph(gate_list):
    """
    Build a directed graph where:
      - Each node is a gate (with its type and qubits)
      - An edge u -> v exists if gate u and v act on any common qubit, and u occurs before v
    """
    G = nx.DiGraph()
    
    for idx, (gate_name, qubits) in enumerate(gate_list):
        G.add_node(idx, gate=gate_name, qubits=qubits)

    # Add edges based on shared qubit and order
    for i in range(len(gate_list)):
        for j in range(i+1, len(gate_list)):
            qubit_i = set(gate_list[i][1])
            qubit_j = set(gate_list[j][1])
            if qubit_i & qubit_j:
                G.add_edge(i, j)

    return G

def visualize_gate_graph(G, max_nodes=20):
    """
    Visualize the first few nodes of the graph (for debugging only).
    """
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    for i, (node, attr) in enumerate(G.nodes(data=True)):
        if i >= max_nodes:
            break
        print(f"Node {node}: gate={attr['gate']}, qubits={attr['qubits']}")
    print("Sample edges:")
    print(list(G.edges())[:10])

if __name__ == "__main__":
    from qasm_parser import load_qasm_file

    # Example usage
    path = "../origin_circuit/qubits_3/grover_3.qasm"
    gate_list, _ = load_qasm_file(path)
    G = build_gate_interaction_graph(gate_list)
    visualize_gate_graph(G)
