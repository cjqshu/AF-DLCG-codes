import math

def get_node_positions(nodes_list, radius=6):

    num_nodes = len(nodes_list)
    radius = radius

    longest_length = max(len(s) for s in nodes_list)

    output_str = ""
    for i in range(1, num_nodes + 1):
        x = math.cos((math.pi * 2 / num_nodes) * float(i)) * radius
        y = math.sin((math.pi * 2 / num_nodes) * float(i)) * radius
        line = f"{nodes_list[i-1]:<{longest_length}} [pos=\"{x:.3f},{y:.3f}!\"];\n"
        output_str += line
    return output_str


if __name__ == "__main__":
    nodes_list = ["Mo", "Cr", "Ni", "W", "NaCl", "temperature", "heating_rate", "roughness", "CPT"]
    output_str = get_node_positions(nodes_list, radius=6)
    print(f"The node coordinates: \n\n{output_str}")