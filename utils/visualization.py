import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def bipartite_graph_visualization(n_agents, m_items, preference, allocation):
    # Create a bipartite graph
    G = nx.DiGraph()

    # Define agents and items for labels
    agents = [f"Agent {i+1}" for i in range(n_agents)]
    items = [f"Item {j+1}" for j in range(m_items)]


    # Add agents nodes
    G.add_nodes_from(agents, bipartite=0)

    # Add items nodes
    G.add_nodes_from(items, bipartite=1)

    # Add edges between agents and items based on allocation
    allocation_edge_labels = {}
    valuation_edge_labels = {}

    for i in range(n_agents):
        for j in range(m_items):
            if allocation[i][j] == 1:
                allocation_edge = (items[j], agents[i])  # Edge from item to agent
                G.add_edge(*allocation_edge, weight=preference[i, j])
                # print(f"Agent {i+1} gets Item {j+1} with value {preference[i, j]}")
                allocation_edge_labels[allocation_edge] = f"{preference[i, j]}"
            else:
                valuation_edge = (agents[i], items[j])
                G.add_edge(*valuation_edge, weight=preference[i, j])
                valuation_edge_labels[valuation_edge] = f"{preference[i, j]}"

    # Generate a bipartite layout to separate the nodes into two groups
    pos = nx.bipartite_layout(G, agents)

    # Add edges with different types
    for edges, weight in allocation_edge_labels.items():
        G.add_edges_from(allocation_edge_labels, edge_type='allocation', weight=weight, color='blue')
    for edge, weight in valuation_edge_labels.items():
        G.add_edge(*edge, edge_type='valuation', weight=weight, color='red')

    # Draw the graph nodes
    plt.figure(figsize=(12, 6))  # Set the size of the figure
    nx.draw_networkx_nodes(G, pos, node_color=['skyblue' if node in agents else 'lightgreen' for node in G.nodes()],
                        node_size=500)

    # Draw edges by type
    nx.draw_networkx_edges(G, pos, edgelist=allocation_edge_labels, edge_color='blue', style='solid', arrows=True, arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=list(valuation_edge_labels.keys()), edge_color='red', style='dashed', arrows=True, arrowsize=15)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='black')

    # Draw edge labels for valuations
    nx.draw_networkx_edge_labels(G, pos, edge_labels=allocation_edge_labels, font_color='blue', label_pos=0.4)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=valuation_edge_labels, font_color='red', label_pos=0.4)

    plt.title('Allocation of Items to Agents with Valuations')
    plt.show()