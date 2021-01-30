from pyspark import SparkConf, SparkContext

def map_line(line):
    # make src, dest pair
    wordlist = line.split()
    return (wordlist[0], wordlist[1])

def map_contrib_pairs(row):
    pair_list = []
    dests, rank = row[1][0], row[1][1]
    degree = len(dests)
    if degree > 0:
        for dest in dests:
            pair_list.append((dest, 0.8*(rank / degree)))
        
    return pair_list
    
def map_add_back(p):
    a ,b = p
    if a != None and b != None:
        return max(a,b)
    elif a == None and b != None:
        return b
    else:
        return a


def compute_pagerank(sc, nodes_set, N, in_file, iterations=20):
    
    rdd = sc.textFile(in_file).map(map_line)
    # make (source, ([dest1, dest2, ..]) pair
    links = rdd.groupByKey()
    
    # init. rank values by 1/N
    ranks = links.map(lambda x: (x[0], 1.0/N))
    
    all_nodes = sc.parallelize(nodes_set).map(lambda x: (x,0.0))
    
    for _ in range(iterations):
        contribs = links.join(ranks).flatMap(map_contrib_pairs)
        
        new_ranks = contribs.reduceByKey(lambda x,y: x+y).cache()
        
        # add back nodes that in-degree = 0 (they will not appear in contribs 
        # since they will not be in any dests), and set their ranks to 0
        new_ranks = all_nodes.leftOuterJoin(new_ranks).mapValues(map_add_back)
        #print(new_ranks.collect())
        
        S = new_ranks.map(lambda x: x[1]).reduce(lambda x,y: x+y)
        #print(f"S = {S}")
        
        # Re-insert leaked Pagerank
        ranks = new_ranks.mapValues(lambda rk: rk + (1-S)/N).cache()

    return ranks.sortBy(lambda x: -x[1])    # decreasing order


def get_N(in_file):
    nodes = set()
    with open(in_file, 'r') as f:
        for line in f.readlines():
            src, dest = line.split()
            if src not in nodes:
                nodes.add(src)
            if dest not in nodes:
                nodes.add(dest)
    return nodes, len(nodes)


if __name__ == '__main__':
    conf = SparkConf().setMaster("local").setAppName("pyspark-PageRank")
    sc = SparkContext.getOrCreate(conf=conf)

    in_file = "p2p-Gnutella04.txt"
    #in_file = "test.txt"

    nodes_set, N = get_N(in_file)
    print("nodes: ", N)
    ranks = compute_pagerank(sc, nodes_set, N, in_file)

    # write output
    out_file = "./Outputfile.txt"
    with open(out_file,'w+') as f:
        for e in ranks.take(10):
            print("{},    {:.8f}".format(e[0], float(e[1])))
            f.write("{},    {:.8f}\n".format(e[0], float(e[1])) )

