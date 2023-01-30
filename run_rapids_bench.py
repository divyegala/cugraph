import cudf
from cugraph.tree.minimum_spanning_tree import minimum_spanning_tree
import cugraph
import sys

def main():
    fpath = sys.argv[1]
    high_precisions_datasets = ("USA-road-d.NE.gr", "USA-road-d.CAL.gr", "USA-road-d.LKS.gr", "USA-road-d.W.gr", "USA-road-d.CTR.gr", "USA-road-d.USA.gr")
    weight_dtype = 'float64' if fpath.split("/")[2] in high_precisions_datasets else 'float32'
    df = cudf.read_csv(fpath, sep=' ', skiprows=7, names=['discard', 'src', 'dst', 'weight'], usecols=['src', 'dst', 'weight'], dtype=['str', 'int32', 'int32', weight_dtype])

    print(df.head())
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
    ret = minimum_spanning_tree(G)

if __name__ == "__main__":
    main()