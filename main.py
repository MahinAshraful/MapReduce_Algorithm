from pyspark import SparkContext, SparkConf

# ============================================
# Q1: Count Genes and Diseases per Drug
# ============================================


def map_q1(line):
    parts = line.split("\t")
    if len(parts) == 3:
        source, metaedge, target = parts
        if source.startswith("Compound::") and target.startswith("Gene::"):
            if metaedge in ["CuG", "CdG"]:
                return [(source, "gene")]
        elif source.startswith("Compound::") and target.startswith("Disease::"):
            if metaedge in ["CtD", "CpD"]:
                return [(source, "disease")]
    return []


def format_q1(item):
    compound, type_counts = item
    genes = 0
    diseases = 0
    for entity_type, count in type_counts:
        if entity_type == "gene":
            genes = count
        elif entity_type == "disease":
            diseases = count
    return (compound, genes, diseases)


def run_q1(sc):
    edges = sc.textFile("e.tsv")
    header = edges.first()
    edges = edges.filter(lambda line: line != header)

    mapped = edges.flatMap(map_q1)
    counts = mapped.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
    by_compound = counts.map(
        lambda item: (item[0][0], (item[0][1], item[1]))
    ).groupByKey()
    results = by_compound.map(format_q1)
    sorted_results = results.sortBy(lambda x: x[1], ascending=False).collect()

    print("\nQ1 Results:")
    print(f"{'Compound':<30} {'Genes':<10} {'Diseases':<10}")
    for compound, genes, diseases in sorted_results[:5]:
        print(f"{compound:<30} {genes:<10} {diseases:<10}")

    return sorted_results


# ============================================
# Q2: Count Disease Distribution
# ============================================


def map_q2(line):
    parts = line.split("\t")
    if len(parts) == 3:
        source, metaedge, target = parts
        if source.startswith("Compound::") and target.startswith("Disease::"):
            if metaedge in ["CtD", "CpD"]:
                return [(target, source)]
    return []


def run_q2(sc):
    edges = sc.textFile("e.tsv")
    header = edges.first()
    edges = edges.filter(lambda line: line != header)

    pairs = edges.flatMap(map_q2)
    per_disease = pairs.groupByKey().map(lambda x: (x[0], len(set(x[1]))))
    distribution = per_disease.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)
    sorted_dist = distribution.sortBy(lambda x: x[1], ascending=False).collect()

    print("\nQ2 Results:")
    print(f"{'# Compounds':<15} {'# Diseases':<15}")
    for count, num_diseases in sorted_dist[:5]:
        print(f"{count:<15} {num_diseases:<15}")

    return sorted_dist


# ============================================
# Q3: Get Compound Names
# ============================================


def parse_node(line):
    parts = line.split("\t")
    if len(parts) == 3:
        node_id, name, kind = parts
        if kind == "Compound":
            return [(node_id, name)]
    return []


def run_q3(sc, q1_results):
    nodes = sc.textFile("n.tsv")
    header = nodes.first()
    nodes = nodes.filter(lambda line: line != header)

    names = dict(nodes.flatMap(parse_node).collect())

    results = []
    for compound, genes, diseases in q1_results:
        name = names.get(compound, compound)
        results.append((name, genes))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nQ3 Results:")
    print(f"{'Compound Name':<40} {'Gene Count':<10}")
    for name, genes in results[:5]:
        print(f"{name:<40} {genes:<10}")

    return results


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    conf = SparkConf().setAppName("HetIONet").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    q1_results = run_q1(sc)
    q2_results = run_q2(sc)
    q3_results = run_q3(sc, q1_results)

    sc.stop()
