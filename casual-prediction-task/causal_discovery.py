import numpy as np
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM, Notears, DirectLiNGAM, GAE, ANMNonlinear, DAG_GNN
import networkx as nx


class DAGCreator:
    def __init__(self, train_df, X_features, Y_feature, max_samples=1000, method="notears", ) -> None:
        self.train_df = train_df
        self.X_features = X_features
        self.Y_feature = Y_feature
        self.max_samples = max_samples
        self.method = method
        self.predefined_dags = {
            "cpt": self.load_predefined_cpt_CGfusion,
        }

    def run_gcastle(self, method):
        algo_dict = {
            "pc": PC,
            "ges": GES,
            "icalingam": ICALiNGAM,
            "golem": GOLEM,
            "notears": Notears,
            "directlingam": DirectLiNGAM,
            "gae": GAE,
            "anm": ANMNonlinear,
            "dag_gnn": DAG_GNN
        }
        algo = algo_dict[method]()
        algo.learn(self.train_df[self.X_features + self.Y_feature].sample(
                np.min([self.train_df.shape[0], self.max_samples]), random_state=42
            )
        )

        # Relabel the nodes
        learned_graph = nx.DiGraph(algo.causal_matrix)
        MAPPING = {
            k: v for k, v in zip(range(algo.causal_matrix.shape[0]), self.X_features + self.Y_feature)
        }
        learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)
        self.dag_edges = list(learned_graph.edges())

    def remove_target_from_dag_edges(self):
        new_edges = []
        for l in self.dag_edges:
            new_edge = []

            for i in l:
                if i != self.Y_feature[0]:
                    new_edge.append(i)

            # if len(new_edge) >= 2:
            new_edges.append(new_edge)

        self.dag_edges = new_edges

    def add_nodes_not_in_edges(self):
        nodes_in_edges = [item for sublist in self.dag_edges for item in sublist]
        nodes_not_in_edges = [x for x in self.X_features if x not in nodes_in_edges]
        self.dag_edges.append(nodes_not_in_edges)

    def create_dag_constraints(self):
        def get_parents(dag, node):
            # Added childs
            parents = []
            for origin, destination in dag:
                if destination == node:
                    parents.append(origin)
            return parents

        def get_ancestors(dag, node):
            ancestors = []
            for origin, destination in dag:

                # Dont take into account ancestors that go through Y
                if destination == self.Y_feature[0]:
                    continue

                if destination == node:
                    ancestors.append(origin)
                    ancestors.extend(get_ancestors(dag, origin))
            return ancestors

        dag_edges = self.dag_edges
        target_parents = get_parents(dag_edges, self.Y_feature[0])
        dag_constraints = { parent: get_ancestors(dag_edges, parent) + [parent] for parent in target_parents }
        dag_constraints["parents"] = target_parents

        self.dag_constraints = dag_constraints
        self.dag_edges = list(dag_constraints.values())

    def create_dag_edges(self, add_nodes_not_in_edges=False):

        if self.method in ["pc", "ges", "icalingam", "golem", "notears", "gae", "anm"]:
            self.run_gcastle(self.method)

        else:
            print("Method not supported")

        self.create_dag_constraints()
        self.remove_target_from_dag_edges()
        if add_nodes_not_in_edges:
            self.add_nodes_not_in_edges()

        return [list(set(d)) for d in self.dag_edges if len(d) > 0]

    def return_predefined_dag(self, dataset_name):
        return self.predefined_dags[dataset_name]()

    @staticmethod
    def load_predefined_cpt_CGfusion():
        dag = [['Mo_x2', 'Si_x6', 'P_x9', 'N_x11', 'Cr_x16', 'NaCl_x24'],
               ['Si_x6', 'Mo_x2'],
               ['P_x9', 'Mo_x2', 'Cr_x16'],
               ['N_x11', 'Mo_x2', 'Ni_x3', 'PP_x22', 'NaCl_x24'],
              ]
        return dag
