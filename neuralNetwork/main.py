from NNModel import *
from LayersWeightsTesting import *
from tabulate import tabulate
from operator import itemgetter


def build_layers(ontology, tableName):
    LB = layersBuilder(ontology, tableName)
    number_of_layers, layers_dims, layers = LB.build_layers()
    return LB, number_of_layers, layers_dims, layers


def prediction_test(test_x, test_y, test_id, parameters):
    cc, ch, hh, hc, result = model.predict(test_x, test_y, test_id, parameters, print_result=True)
    print("Prediction accuracy on test set = ", result, "%")
    print("hh=", hh)
    print("hc=", hc)
    print("ch=", ch)
    print("cc=", cc)


def inputs_weights_preprocessing(parameters, number_of_layers):
    layers = []
    id_layers = []
    i = 6
    while i > 0:
        weights = parameters['W' + str(i)]
        layer_weights = [sum(x) for x in zip(*weights)]
        layers.append(layer_weights)
        id_layers.append(parameters['IdW' + str(i)])
        i -= 1
    return layers, id_layers


def inputs_weights_calculation(layers, id_layers):
    print(layers)
    print("before--------------------")
    for l in layers:
        print(l[0])
    for layer_index in range(1, len(layers)):

        for weight_index in range(0, len(layers[layer_index])):
            ids = [id_layers[layer_index][i][weight_index] for i in range(len(layers[layer_index-1]))]
            for index_id, id in enumerate(ids):
                if id != 0:
                    layers[layer_index][weight_index] *= layers[layer_index-1][index_id]
                    break
    print("after----------------")
    for l in layers:
        print(l[0])
    print(layers)
    print(layers[-1])
    return layers[-1]


def inputs_weights():
    layer2 = [3, 7, 1, 8]
    id_layer2 = [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    layer3 = [4, 5, 8]
    id_layer3 = [[1, 0, 0], [0, 1, 1]]

    layer4 = [4, 3]
    id_layer4 = [[1, 1]]

    layers = [layer4, layer3, layer2]
    id_layers = [id_layer4, id_layer3, id_layer2]

    for layer_index in range(1, len(layers)):
        for weight_index in range(0, len(layers[layer_index])):
            ids = [id_layers[layer_index][i][weight_index] for i in range(len(layers[layer_index-1]))]
            for index_id, id in enumerate(ids):
                if id != 0:
                    layers[layer_index][weight_index] *= layers[layer_index-1][index_id]
                    break
    print(layers[2])


if __name__ == '__main__':
    #inputs_weights()
    ontology = "crm_bnf.owl"
    tableName = "nn_input"
    LB, number_of_layers, layers_dims, layers = build_layers(ontology, tableName)
    model = NNModel()
    train_id, test_id, train_x, train_y, test_x, test_y = \
        model.load_data(tableName, hors_dusage_Percentage=30, trainpercentage=70)# -1 to load all the dataset
    parameters = model.L_layer_model(train_x, train_y, layers, LB)
    #AL = model.L_model_forward(train_x, parameters)[0]
    #prediction_test(test_x, test_y, test_id, parameters)
    results = percentage_layer_uris_per_class(layers[0], layers[0], train_x.T, train_y.T)



    layerss, id_layers = inputs_weights_preprocessing(parameters, number_of_layers)

    weights = inputs_weights_calculation(layerss, id_layers)

    for i in range(len(weights)):
        weights[i] = abs(weights[i])

    max_weight = max(weights)

    for i in range(len(weights)):
        weights[i] = weights[i]*100/max_weight

    data = []
    for index, URI in enumerate(layers[0]):
        if "pass" not in URI:
            percent_com = (results[URI + '-communicable-exist'] / (results[URI + '-communicable-notExist']+results[URI + '-communicable-exist']))*100
            percent_out = (results[URI + '-out-exist'] / (results[URI + '-out-notExist'] + results[URI + '-out-exist'])) * 100

            info_com = str(results[URI + '-communicable-exist'])+"/"+str(results[URI + '-communicable-exist'] + results[URI + '-communicable-notExist'])
            info_out = str(results[URI + '-out-exist']) + "/" + str(
                results[URI + '-out-exist'] + results[URI + '-out-notExist'])
            data.append([index + 1, URI, percent_com, percent_out, abs(percent_out - percent_com), info_com, info_out, weights[index]])
    print(tabulate(sorted(data, key=itemgetter(5), reverse=True), headers=["index", "URI", "Percent communicable %", "Percent Out-of-order %", "difference distribution", "occurences communicable", "occurences out-of-order", "Weight"]))
    # p=plotRemote()
    """OQ = ontologyQuerying(ontology)
    DT = dataTransformation()
    OQ.node_extraction()
    OQ.edges_extraction()
    nodes = OQ.get_nodes()
    edges = OQ.get_edges()
    edgeDF = DT.edges_to_edgeDF(edges)
    nodeDF = DT.node_to_nodeDF(nodes)
    p = plotRemote(edge_df=edgeDF, node_df=nodeDF)"""