import threading
import time
import os
import rdflib
from layersClass import *
from rdflib.plugins.sparql import prepareQuery
from rdflib import Graph
from sklearn.cluster import DBSCAN
import matplotlib.animation as animation
from matplotlib import style
import pickle
import psycopg2 as psycopg2
import time
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
import numpy as np

class layersBuilder:
    def __init__(self, ontology, tableName):
        self.ontology = ontology
        self.tableName = tableName


    def build_layers(self):
        attributesID=self.get_attributesID()[1: -3] # The last three attributes in the table are the label, the length and the communication
        print('Get the attributes names from the database....')
        time.sleep(2)
        print(len(attributesID)," attributes extracted")
        print("attributes : ",attributesID)
        print("-" * 200)

        leaf_nodes = self.extract_leaf_nodes()  # leaf_nodes["id"]= node url
        ordered_leaf_nodes = self.order_leaf_nodes(attributesID, leaf_nodes)
        print('Create the leaf_nodes["id"] = node url, ', len(ordered_leaf_nodes), ' leaf nodes')


        root_node = self.extract_root_node()
        print('Get the root node : ', root_node)

        number_of_layers = self.extract_length_of_ontology(ordered_leaf_nodes, root_node)+1
        print('Get the number of layers : ', number_of_layers)

        filesize = os.path.getsize("layers.txt")
        if filesize==0:
            print("Create network layers...")
            layers_dims, layers = self.create_layers(ordered_leaf_nodes, number_of_layers, root_node)
            self.save_layers(layers_dims, layers)
        else:
            print("Load network layers...")
            layers_dims, layers = self.get_layers()

        self.print_layers(layers_dims, layers)

        return len(layers), layers_dims, layers


    def add_parent_next_layer(self, layers, index, node, root, number_of_layers):
        if node == root:
            return True

        parent = self.node_parent(node)
        parent_distance_to_root = self.distance_to_root(parent, root)
        parent_position = number_of_layers - 1 - parent_distance_to_root

        if parent_position != index:
            layers[index].append(node+"$$pass")
            self.add_parent_next_layer(layers, index+1, node, root, number_of_layers)
        else:
            if parent not in layers[index]:
                layers[index].append(parent)
                self.add_parent_next_layer(layers, index + 1, parent, root, number_of_layers)


    def print_layers(self, layers_dims, layers):
        for layer in layers:
            print(layer)
        print(layers_dims)

    def get_layers(self, fileName="layers.txt"):
        with open(fileName,"rb") as MyFile:
            layersObject=pickle.load(MyFile)
            return layersObject.NNdimensions, layersObject.NNlayers

    def create_layers(self, leaf_nodes, number_of_layers,root):

        layers = [leaf_nodes]
        for l in range(number_of_layers-1):
            layer = []
            layers.append(layer)

        for index, node in enumerate(layers[0]):
            self.add_parent_next_layer(layers, 1, node, root, number_of_layers)

        layers_dims = []
        for layer in layers:
            layers_dims.append(len(layer))

        return layers_dims, layers

    def save_layers(self, layers_dims, layers):
        with open("layers.txt", "wb") as MyFile:
            layersObject=layersClass(layers, layers_dims)
            pickle.dump(layersObject, MyFile)

    def distance_to_root(self, node, root):
        if node == root:
            return 0

        graph = rdflib.Graph()
        graph.parse(self.ontology)

        distance = 0
        q = prepareQuery('''select (count(?mid) as ?distance) { 
                <''' + node + '''> rdfs:subClassOf* ?mid .
                ?mid rdfs:subClassOf+ <''' + root + '''> .
                }''')
        for row in graph.query(q):
            distance = row[0]

        return int(distance)


    def node_parent(self, node):
        graph = rdflib.Graph()
        graph.parse(self.ontology)

        parent = ""
        q = prepareQuery('''select ?parent { 
                    <''' + node + '''> rdfs:subClassOf ?parent.
                    }''')
        for row in graph.query(q):
            parent = str(row[0])

        return str(parent)


    def extract_length_of_ontology(self, leaf_nodes, root):
        """
        Arguments:
        leaf_nodes -- python array (list) containing the URL of the leaf nodes
        root -- URL of the root node

        Returns:
        parameters -- python integer (int) containing the maximal distance between the leaf nodes and the root
        """
        graph = rdflib.Graph()
        graph.parse(self.ontology)
        max = 0

        for leaf_node in leaf_nodes:
            q = prepareQuery('''select (count(?mid) as ?distance) { 
            <''' + leaf_node + '''> rdfs:subClassOf* ?mid .
            ?mid rdfs:subClassOf+ <''' + root + '''> .
            }''')
            for row in graph.query(q):
                if int(row[0]) > max:
                    max = int(row[0])
        return max


    def extract_root_node(self):

        graph = rdflib.Graph()
        graph.parse(self.ontology)

        q = prepareQuery('''SELECT ?a WHERE { 
                            ?a rdf:type <http://www.w3.org/2002/07/owl#Class>.
                            FILTER NOT EXISTS { ?a rdfs:subClassOf ?b}
                        }''')
        root_node = ""
        for row in graph.query(q):
            root_node = row[0]
        return str(root_node)


    def order_leaf_nodes(self,attributesID, leaf_nodes):
        ordered_leaf_nodes = []
        for id in attributesID:
            if str(id) in leaf_nodes.keys():
                ordered_leaf_nodes.append(leaf_nodes[str(id)])
        return ordered_leaf_nodes


    def extract_leaf_nodes(self):
        graph = rdflib.Graph()
        graph.parse(self.ontology)

        q = prepareQuery('''SELECT ?id ?a WHERE { ?a <http://www.semanticweb.org/CrmBnF#Id> ?id.
                        FILTER NOT EXISTS { ?b rdfs:subClassOf ?a}
                        }''')
        leaf_nodes = {}
        for row in graph.query(q):
            s = str(row[0])
            leaf_nodes[s] = str(row[1])


        return leaf_nodes


    def get_attributesID(self):
        conn = psycopg2.connect(
            "dbname=BNF port=5432 user=postgres password=Postalaa1")
        cursor = conn.cursor()

        query = "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '" + self.tableName + "'"
        cursor.execute(query)

        attributesNames = []  # the attributes' names are the leafs ids in the ontology

        for row in cursor:
            attributesNames.append(row[0].upper())
        return attributesNames