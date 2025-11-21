from pyspark import SparkContext, SparkConf


# helper: load node names
def parse_node(line):
    """Extracts (ID, Name) from nodes.tsv"""
    line = line.strip()  # remove whitespace
    parts = line.split("\t")
    if len(parts) == 3:
        node_id, name, kind = parts
        if kind.strip() == "Compound":
            return [(node_id.strip(), name.strip())]
    return []


def load_names(sc):
    """Creates a dictionary of Drug ID -> Drug Name"""
    nodes = sc.textFile("nodes.tsv")
    header = nodes.first()
    nodes = nodes.filter(lambda line: line != header)  # skip header

    # collect results into a Python Dictionary for fast lookup
    names_rdd = nodes.flatMap(parse_node)
    return dict(names_rdd.collect())


# Q1: Count Genes and Diseases per Drug


def map_q1(line):
    parts = line.split("\t")
    if len(parts) == 3:
        source, metaedge, target = parts

        # check for drug -> gene interactions
        if source.startswith("Compound::") and target.startswith("Gene::"):
            if metaedge in ["CuG", "CdG", "CbG"]:
                return [(source, ("gene", target))]

        # Check for drug -> disease interactions
        elif source.startswith("Compound::") and target.startswith("Disease::"):
            if metaedge in ["CtD", "CpD"]:
                return [(source, ("disease", target))]
    return []


def run_q1(sc, names_dict):
    edges = sc.textFile("edges.tsv")
    header = edges.first()
    edges = edges.filter(lambda line: line != header)  # remove header

    # 1. Map edges to (DrugID, (Type, TargetID))
    mapped = edges.flatMap(map_q1)

    # 2. Group all targets by DrugID
    grouped = mapped.groupByKey()

    # 3. Reducer: Count unique targets
    def count_distinct_entities(item):
        compound, values = item
        genes = set()  # Use Set to remove duplicates automatically
        diseases = set()

        for entity_type, target_id in values:
            if entity_type == "gene":
                genes.add(target_id)
            elif entity_type == "disease":
                diseases.add(target_id)

        return (compound, len(genes), len(diseases))

    results = grouped.map(count_distinct_entities)

    # 4. Sort by Gene Count (High to Low)
    sorted_results = results.sortBy(lambda x: x[1], ascending=False).collect()

    print("\n" + " " * 40)
    print("Q1 Results (Top 5)")
    print(" " * 40)
    print(f"{'Compound Name':<30} {'Genes':<10} {'Diseases':<10}")

    for compound, genes, diseases in sorted_results[:5]:
        # Look up the name in the dictionary we loaded earlier
        clean_id = compound.strip()
        display_name = names_dict.get(clean_id, clean_id)
        print(f"{display_name:<30} {genes:<10} {diseases:<10}")

    return sorted_results


# Q2: Count Disease Distribution


def map_q2(line):
    parts = line.split("\t")
    if len(parts) == 3:
        source, metaedge, target = parts
        # We want (Disease, Drug) to count drugs per disease
        if source.startswith("Compound::") and target.startswith("Disease::"):
            if metaedge in ["CtD", "CpD"]:
                return [(target, source)]  # Key = Disease, Value = Drug
    return []


def run_q2(sc):
    edges = sc.textFile("edges.tsv")
    header = edges.first()
    edges = edges.filter(lambda line: line != header)

    pairs = edges.flatMap(map_q2)

    # 1. Count unique drugs per disease
    # Result: (DiseaseA, 10 drugs), (DiseaseB, 5 drugs)...
    per_disease = pairs.groupByKey().map(lambda x: (x[0], len(set(x[1]))))

    # 2. Create a Histogram (Distribution)
    # Map (10 drugs, 1) -> ReduceByKey (Sum)
    # Result: 5 diseases have 10 drugs associated
    distribution = per_disease.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)

    sorted_dist = distribution.sortBy(lambda x: x[1], ascending=False).collect()

    print("\n" + " " * 40)
    print("Q2 Results (Top 5)")
    print(" " * 40)
    print(f"{'# Compounds':<15} {'# Diseases':<15}")
    for count, num_diseases in sorted_dist[:5]:
        print(f"{count:<15} {num_diseases:<15}")

    return sorted_dist


# Q3: Get Compound Names (Top 5 Genes)


def run_q3(sc, q1_results, names_dict):
    # Q3 is essentially re-formatting Q1 results with names
    results = []
    for compound_id, genes, diseases in q1_results:
        clean_id = compound_id.strip()
        name = names_dict.get(clean_id, clean_id)  # Name Lookup
        results.append((name, genes))

    # Sort Descending by Gene Count
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + " " * 40)
    print("Q3 Results (Top 5)")
    print(" " * 40)
    print(f"{'Compound Name':<40} {'Gene Count':<10}")
    for name, genes in results[:5]:
        print(f"{name:<40} {genes:<10}")

    return results


if __name__ == "__main__":
    # Spark Setup
    conf = SparkConf().setAppName("HetIONet").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    print("Starting Spark Job...")

    # 1. Load Names into memory first (Optimization)
    names_dict = load_names(sc)

    # 2. Run Q1 (Pass names for printing)
    q1_output = run_q1(sc, names_dict)

    # 3. Run Q2
    q2_output = run_q2(sc)

    # 4. Run Q3
    q3_output = run_q3(sc, q1_output, names_dict)

    print("\nJob Finished.")
    sc.stop()
