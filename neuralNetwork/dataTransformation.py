import pandas as pd
import numpy as np
class dataTransformation:
    def __init__(self):
        pass

    def edges_to_edgeDF(self, edges):
        edge_df = pd.DataFrame(np.array(edges),columns=['from', 'to', 'relationship'])
        return edge_df

    def node_to_nodeDF(self, nodes):
        node_df = pd.DataFrame(np.array(nodes), columns=['id', 'type'])
        return node_df