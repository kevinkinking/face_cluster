from random import shuffle
import networkx as nx
import numpy as np
import config
from face_feature import lcnn_feature
import os

def chinese_whispers_cluster(encoding_list, threshold=0.30, iterations=20):
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx + 1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx], 
            'encoding': face_encoding_to_check})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        similars = lcnn_feature.feature_similar_s(face_encoding_to_check, compare_encodings)
        #print similars
        encoding_edges = []
        for i, similar in enumerate(similars):
            if similar > threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight':similar - threshold}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            #get all neighbor's id,the id always is int type
            neighbors = G[node]
            clusters = {}
            #get all neighbor edge's weight int to hashmap
            #print len(neighbors)
            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        #print 'saved into clusters on the before'
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        #print 'saved into clusters'
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']
        encoding = np.array(data['encoding'])
        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append([path,encoding])

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def get_face_features_dic(face_img_dir):
	face_img_names = os.listdir(face_img_dir)
    face_features_dic = {}
    for face_img_name in face_img_names:
        face_img_path = os.path.join(face_img_dir, img_name)
        face_img = cv2.imread(face_img_path)
        face_feature = lcnn_feature.feature_extract(face_img)
        face_features_dic[face_img_path] = face_feature
    return face_features_dic

if __name__ == '__main__':
	lcnn_feature.init_model(config.lcnn_face_prototxt, config.lcnn_face_caffemodel)
	face_features_dic = get_face_features_dic('./data/imgs_test')
	clusted_faces_dic = chinese_whispers_cluster(face_features_dic)
	print clusted_faces_dic