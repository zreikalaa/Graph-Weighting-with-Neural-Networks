from jaal import Jaal
from jaal.datasets import load_got
import pandas as pd
class plotRemote:


    def __init__(self, edge_df, node_df):
        #edge_df, node_df = load_got()
        Jaal(edge_df,node_df).plot()
        """edge_data=[{'from': 1, 'to': 2, 'type': 'directed', 'weight': 5, 'strength': 'high'},
                   {'from': 1, 'to': 3, 'weight': 2, 'strength': 'medium'}]
        edge_df = pd.DataFrame(edge_data)

        node_data = [{'id': 1, 'gender': 'female'},
                     {'id': 1, 'gender': 'male'},
                     {'id': 3, 'gender': 'male'}]
        node_df=pd.DataFrame(node_data)"""