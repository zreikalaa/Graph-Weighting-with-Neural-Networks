import rdflib
from rdflib.plugins.sparql import prepareQuery


class ontologyQuerying:

    def __init__(self, ontology):
        self.nodes = []
        self.edges = []
        self.ontology = ontology

    def node_extraction(self):
        """create the nodes list, type : 2D list
        ex: [[t1, treatment], [d1, degradation]..., [événement, event]] """

        graph = rdflib.Graph()
        graph.parse(self.ontology)
        treatments = []

        """Get the nodes that are subClassOf* treatment i.e. type=treatment"""
        q = prepareQuery('''select ?class
         where { ?class rdfs:subClassOf+ <http://www.semanticweb.org/CrmBnF#Traitement>}''')

        for row in graph.query(q):
            treatments.append([str(row[0]), 'treatment'])

        """Get the nodes that are subClassOf* degradation i.e. type=degradation"""
        q = prepareQuery('''select ?class
                where { ?class rdfs:subClassOf+ <http://www.semanticweb.org/CrmBnF#Dégradation>}''')
        degradations = []
        for row in graph.query(q):
            degradations.append([str(row[0]), 'degradation'])

        """Get the three high level concepts"""
        q = prepareQuery('''select ?class
                        where { ?class rdfs:subClassOf <http://www.semanticweb.org/CrmBnF#événement>}''')
        nodes = []
        nodes.extend(treatments)
        nodes.extend(degradations)
        for row in graph.query(q):
            if str(row[0]).split('#')[1] == "Communication":
                nodes.append([str(row[0]), 'communication'])
            elif str(row[0]).split('#')[1] == "Traitement":
                nodes.append([str(row[0]), 'treatment'])
            else:
                nodes.append([str(row[0]), 'degradation'])

        nodes.append(("événement", "event"))
        self.nodes = nodes

    def edges_extraction(self):
        """create the edges 2D list, type : array of triples.
                structure : [[from, to, relationship], [], ..., []] """

        graph = rdflib.Graph()
        graph.parse(self.ontology)
        edges = []

        """Get the subClassOf edges"""
        q = prepareQuery('''select ?s ?o
                 where { ?s rdfs:subClassOf ?o}''')
        for row in graph.query(q):
            edges.append([str(row[0]), str(row[1]), "subClassOf"])

        """Get the equivalentClass edges"""
        q = prepareQuery('''select ?s ?o
                         where { ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?o}''')
        for row in graph.query(q):
            edges.append([str(row[0]), str(row[1]), "equivalentClass"])

        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges
