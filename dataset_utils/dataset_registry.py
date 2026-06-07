"""Dataset registry utilities for node-classification tasks."""

from __future__ import annotations

from typing import Dict, List


def _summarize_graph_dataset(
    dataset_name: str,
    dataset,
    num_features: int,
    num_classes: int,
) -> None:
    """Print aggregate statistics for loaded graph level task datasets."""
    try:
        num_graphs = len(dataset)
    except TypeError:
        num_graphs = 0

    total_nodes = 0
    try:
        for data in dataset:
            node_count = getattr(data, "num_nodes", None)
            if node_count is None:
                if hasattr(data, "x") and data.x is not None:
                    node_count = data.x.size(0)
                elif hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
                    node_count = int(data.edge_index.max().item()) + 1
                else:
                    node_count = 0
            total_nodes += int(node_count)
    except Exception:
        total_nodes = 0

    print(
        f"[GraphDataset] {dataset_name}: graphs={num_graphs}, nodes={total_nodes}, "
        f"features={num_features}, classes={num_classes}"
    )


class BaseDatasetRegistry:
    """Basic registry providing shared helpers for dataset name listing."""

    data_sources: Dict[str, Dict[str, object]]

    def __init__(self) -> None:
        self.data_sources = {}

    def _combine_keys(self) -> List[str]:
        dataset_list: List[str] = []
        for group in self.data_sources.values():
            dataset_list.extend(group.keys())
        return dataset_list

    @staticmethod
    def _sort_key(dataset_name: str):
        parts = dataset_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            return (int(parts[0]), parts[1])
        return (float('inf'), dataset_name)

    def get_dataset_list(self) -> List[str]:
        combined = self._combine_keys()
        return sorted(combined, key=self._sort_key)

    # Backward compatibility for legacy callers of this method name.
    def get_node_level_dataset_list(self) -> List[str]:
        return self.get_dataset_list()


class NodeLevelDatasetRegistry(BaseDatasetRegistry):
    """Registry for Node4All node-classification transfer benchmarks."""

    def __init__(self) -> None:
        super().__init__()
        self.data_sources = {
            "dgl": {},
            "ogb": {
                "27_ogbn_arxiv": "ogbn-arxiv",
                #"28_ogbn_products": "ogbn-products",
            },
            "heterophilous": {
                "21_texas": "texas_4_classes",
                "13_cornell": "cornell",
                "24_wisconsin": "wisconsin",
                "8_chameleon": "chameleon_filtered",
                "20_squirrel": "squirrel_filtered",
                "19_roman_empire": "roman_empire",
                "6_amazon_ratings": "amazon_ratings",
                "22_tolokers": "tolokers",
                "16_minesweeper": "minesweeper",
                "18_questions": "questions",
                "0_actor": "actor",
            },
            "pyg": {
                '12_cora': {"class": "Planetoid", "name": "Cora"},
                '9_citeseer': {"class": "Planetoid", "name": "CiteSeer"},
                '17_pubmed': {"class": "Planetoid", "name": "PubMed"},
                '25_wiki_cs': {"class": "WikiCS"},
                "26_full_cora": {"class": "CoraFull"},
                '10_co_cs': {"class": "Coauthor", "name": "CS"},
                '11_co_phy': {"class": "Coauthor", "name": "Physics"},
                '4_amz_computer': {"class": "Amazon", "name": "Computers"},
                '5_amz_photo': {"class": "Amazon", "name": "Photo"},
                "14_dblp": {"class": "CitationFull", "name": "DBLP"},
                "23_wiki": {"class": "AttributedGraphDataset", "name": "Wiki"},
                # "1_air_brazil": {"class": "Airports", "name": "Brazil"},
                # "3_air_usa": {"class": "Airports", "name": "USA"},
                # "2_air_eu": {"class": "Airports", "name": "Europe"},
                "7_blogcatalog": {"class": "AttributedGraphDataset", "name": "BlogCatalog"},
                "15_deezer": {"class": "DeezerEurope"},
            },
        }

        self.dgl_datasets = set(self.data_sources["dgl"].keys())
        self.heterophilous_datasets = set(self.data_sources["heterophilous"].keys())
        self.ogb_datasets = set(self.data_sources["ogb"].keys())
        self.pyg_datasets = set(self.data_sources["pyg"].keys())


# Backward compatibility aliases
NodeClassificationDatasetRegistry = NodeLevelDatasetRegistry


def get_node_level_dataset_list() -> List[str]:
    return NodeLevelDatasetRegistry().get_dataset_list()


def get_node_dataset_list() -> List[str]:
    return get_node_level_dataset_list()
