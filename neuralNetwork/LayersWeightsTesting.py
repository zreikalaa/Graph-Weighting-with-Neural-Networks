import rdflib
from rdflib.plugins.sparql import prepareQuery
global_input_layer = []
def leaf_nodes(conceptURI, ontology="crm_bnf.owl"):
    """return an array that contains the URI of the leaf nodes of a concept (owl:class) in an ontology"""
    graph = rdflib.Graph()
    graph.parse(ontology)

    q = prepareQuery('''SELECT ?x WHERE { ?x rdfs:subClassOf+ <''' + conceptURI + '''>.
                                FILTER NOT EXISTS { ?a rdfs:subClassOf ?x}
                                }''')
    graph.query(q)
    URIs = []
    for row in graph.query(q):
        URIs.append(str(row[0]))
    if len(URIs)==0:
        URIs.append(conceptURI)
    return URIs


def position_of_URI(URI):
    """return the position of the node that correspond to the URI in the input layer"""
    global global_input_layer
    return global_input_layer.index(URI)

def position_of_URIS(URIS):
    """return the positions in the input layer for a set of URI"""
    positions = []
    for URI in URIS:
        if URI in global_input_layer:
            positions.append(position_of_URI(URI))
    return positions

def percentage_uri_per_class(URI, train_x, train_y):
    """input:
    URI : the URI to calculate its precentage in the two classes
    train_x : n*m np array where n is the number of the testing elements and m is the number of dimensions (leaf nodes)
    train_y : n*1 np array where train_y[i][0] is the class of the ith element (value = 0 or 1)

    return a dictionary d:
    d['communicable-exist'] : the number of communicable vectors that contain the concept represented by the URI
    d['communicable-notExist'] the number of communicable vectors that do not contain the concept represented by the URI
    d['out-exist'] the number of out-of-order vectors that contain the concept represented by the URI
    d['out-notExist'] the number of out-of-order vectors that do not contain the concept represented by the URI"""
    d = {}
    d['communicable-exist'] = 0
    d['communicable-notExist'] = 0
    d['out-exist'] = 0
    d['out-notExist'] = 0
    leaf_nodes_URI = leaf_nodes(URI)
    leaf_nodes_positions = position_of_URIS(leaf_nodes_URI)
    for item_index, row in enumerate(train_x):
        leafs_values = [row[index] for index in leaf_nodes_positions]
        sum_leafs = sum(leafs_values)
        item_class = train_y[item_index][0]
        if sum_leafs == 0 and train_y[item_index][0] == 0:
            d['out-notExist'] += 1
        elif sum_leafs != 0 and train_y[item_index][0] == 0:
            d['out-exist'] += 1
        elif sum_leafs == 0 and train_y[item_index][0] == 1:
            d['communicable-notExist'] += 1
        elif sum_leafs != 0 and train_y[item_index][0] == 1:
            d['communicable-exist'] += 1
    return d


def percentage_layer_uris_per_class(input_layer, target_layer, train_x, train_y):
    """input:
    train_x : n*m np array where n is the number of the testing elements and m is the number of dimensions (leaf nodes)
    train_y : n*1 np array where train_y[i][0] is the class of the ith element (value = 0 or 1)

    return a dictionary results that for each URI in the layer (layer_index):
    results['URI-communicable-exist'] : the number of communicable vectors that contain the concept represented by the URI
    results['URI-communicable-notExist'] the number of communicable vectors that do not contain the concept represented by the URI
    results['URI-out-exist'] the number of out-of-order vectors that contain the concept represented by the URI
    results['URI-out-notExist'] the number of out-of-order vectors that do not contain the concept represented by the URI
    """
    global global_input_layer
    global_input_layer = input_layer
    results = {}
    for URI in target_layer:
        if "pass" not in URI: #No calculation if the node is a pass node (black node)
            d = percentage_uri_per_class(URI, train_x, train_y)
            results[URI + '-communicable-exist'] = d['communicable-exist']
            results[URI + '-communicable-notExist'] = d['communicable-notExist']
            results[URI + '-out-exist'] = d['out-exist']
            results[URI + '-out-notExist'] = d['out-notExist']

    return results