"""Construct a small Bayesian network and visualize its directed graph."""
import networkx as nx
import matplotlib.pyplot as plt
try:
 from pgmpy.models import DiscreteBayesianNetwork
 from pgmpy.factors.discrete import TabularCPD
 model=DiscreteBayesianNetwork([('Rain','WetGrass'),('Sprinkler','WetGrass')])
 model.add_cpds(TabularCPD('Rain',2,[[.8],[.2]]),TabularCPD('Sprinkler',2,[[.6],[.4]]),TabularCPD('WetGrass',2,[[.99,.2,.1,.01],[.01,.8,.9,.99]],evidence=['Rain','Sprinkler'],evidence_card=[2,2]))
 print('Model valid:',model.check_model()); graph=nx.DiGraph(model.edges())
except ImportError:
 print('Install pgmpy to build CPDs; showing network structure only.'); graph=nx.DiGraph([('Rain','WetGrass'),('Sprinkler','WetGrass')])
nx.draw_networkx(graph,with_labels=True,node_color='lightblue',node_size=2200,arrowsize=20); plt.title('Bayesian network'); plt.axis('off'); plt.show()
