from pyspark import SparkContext, SparkConf

# Q1: Count genes and diseases per compound


def map_q1(line):
    # Split line by tab into source, metaedge, target
    parts = line.split("\t")
    if len(parts) == 3:
        source, metaedge, target = parts

        # Check for compound-gene relationships
        if source.startswith("Compound::") and target.startswith("Gene::"):
            if metaedge in ["CuG", "CdG"]:
                return [(source, "gene")]

        # Check for compound-disease relationships
        elif source.startswith("Compound::") and target.startswith("Disease::"):
            if metaedge in ["CtD", "CpD"]:
                return [(source, "disease")]
    return []


def format_q1(item):
    # Extract compound and its type counts
    compound, type_counts = item
    genes = 0
    diseases = 0

    # Count genes and diseases separately
    for entity_type, count in type_counts:
        if entity_type == "gene":
            genes = count
        elif entity_type == "disease":
            diseases = count

    return (compound, genes, diseases)


def add_count(pair):
    return (pair, 1)


def sum_counts(a, b):
    return a + b


def filter_header(line, header):
    return line != header


def reorganize_by_compound(item):
    return (item[0][0], (item[0][1], item[1]))


def get_gene_count(item):
    return item[1]


def run_q1(sc):
    # Read edges file
    edges = sc.textFile("e.tsv")
    header = edges.first()
    edges = edges.filter(lambda line: line != header)

    # Map: extract compound-gene and compound-disease pairs
    mapped = edges.flatMap(map_q1)

    # Reduce: count occurrences of each pair
    counts = mapped.map(add_count).reduceByKey(sum_counts)

    # Group by compound
    by_compound = counts.map(reorganize_by_compound).groupByKey()

    # Format results
    results = by_compound.map(format_q1)

    # Sort by gene count and collect top 5
    sorted_results = results.sortBy(get_gene_count, ascending=False).collect()

    # Print results
    print("\nQ1 Results:")
    print(f"{'Compound':<30} {'Genes':<10} {'Diseases':<10}")
    for compound, genes, diseases in sorted_results[:5]:
        print(f"{compound:<30} {genes:<10} {diseases:<10}")

    return sorted_results


# Q2: Count disease distribution


def map_q2(line):
    # Split line and extract disease-compound pairs
    parts = line.split("\t")
    if len(parts) == 3:
        source, metaedge, target = parts
        if source.startswith("Compound::") and target.startswith("Disease::"):
            if metaedge in ["CtD", "CpD"]:
                return [(target, source)]
    return []


def count_unique_compounds(item):
    disease, compounds = item
    return (disease, len(set(compounds)))


def flip_to_count(item):
    disease, count = item
    return (count, 1)


def get_disease_count(item):
    return item[1]


def run_q2(sc):
    # Read edges file
    edges = sc.textFile("e.tsv")
    header = edges.first()
    edges = edges.filter(lambda line: line != header)

    # Map: get disease-compound pairs
    pairs = edges.flatMap(map_q2)

    # Count unique compounds per disease
    per_disease = pairs.groupByKey().map(count_unique_compounds)

    # Count how many diseases have each compound count
    distribution = per_disease.map(flip_to_count).reduceByKey(sum_counts)

    # Sort and collect top 5
    sorted_dist = distribution.sortBy(get_disease_count, ascending=False).collect()

    # Print results
    print("\nQ2 Results:")
    print(f"{'# Compounds':<15} {'# Diseases':<15}")
    for count, num_diseases in sorted_dist[:5]:
        print(f"{count:<15} {num_diseases:<15}")

    return sorted_dist


# Q3: Get compound names


def parse_node(line):
    # Extract compound ID and name from nodes file
    parts = line.split("\t")
    if len(parts) == 3:
        node_id, name, kind = parts
        if kind == "Compound":
            return [(node_id, name)]
    return []


def run_q3(sc, q1_results):
    # Read nodes file
    nodes = sc.textFile("n.tsv")
    header = nodes.first()
    nodes = nodes.filter(lambda line: line != header)

    # Create dictionary of compound IDs to names
    names = dict(nodes.flatMap(parse_node).collect())

    # Match compound IDs to names
    results = []
    for compound, genes, diseases in q1_results:
        name = names.get(compound, compound)
        results.append((name, genes))

    # Sort by gene count
    results.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print("\nQ3 Results:")
    print(f"{'Compound Name':<40} {'Gene Count':<10}")
    for name, genes in results[:5]:
        print(f"{name:<40} {genes:<10}")

    return results


# Main function
if __name__ == "__main__":
    # Initialize Spark
    conf = SparkConf().setAppName("HetIONet").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # Run all queries
    q1_results = run_q1(sc)
    q2_results = run_q2(sc)
    q3_results = run_q3(sc, q1_results)

    # Stop Spark
    sc.stop()
