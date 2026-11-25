"""Sample a Bayesian network and infer a posterior with variable elimination."""
try:
 from pgmpy.models import DiscreteBayesianNetwork
 from pgmpy.factors.discrete import TabularCPD
 from pgmpy.inference import VariableElimination
 from pgmpy.sampling import BayesianModelSampling
 model=DiscreteBayesianNetwork([('Rain','WetGrass'),('Sprinkler','WetGrass')])
 model.add_cpds(TabularCPD('Rain',2,[[.8],[.2]]),TabularCPD('Sprinkler',2,[[.6],[.4]]),TabularCPD('WetGrass',2,[[.99,.2,.1,.01],[.01,.8,.9,.99]],evidence=['Rain','Sprinkler'],evidence_card=[2,2]))
 samples=BayesianModelSampling(model).forward_sample(size=5000,show_progress=False); print(samples.head())
 print('P(Rain | WetGrass=1):',VariableElimination(model).query(['Rain'],evidence={'WetGrass':1},show_progress=False))
except ImportError as e: raise SystemExit('Install pgmpy: pip install pgmpy') from e
