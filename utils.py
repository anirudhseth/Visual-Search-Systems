import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

class VocabularyTree():

    def __init__(self,b,depth,features):
        self.b=b    #branching factor of the vocabulary tree
        self.depth=depth    #depth of the vocabulary tree
        self.root=features.mean(axis = 0)   #centroid at the root is mean of all desciptors dim=(128)
        self.nodes={}   #dictionary for nodes of the tree , holds the centroid each has dim =(128)
        self.nodes[0] = self.root   #assign starting node as the root
        self.currentnode=0    #current node to keep track when building the tree
        self.tree={}    # dictionary for the tree data structure eg : 0: [1, 22, 43, 64] , parent : [child1,child2,..]
        self.imageFreqInLeaves={}  # dictionary with count of image features for a leaf node eg: '3':{'obj10': 13,'obj11': 58,'obj12': 66,..}
        self.model = MiniBatchKMeans(n_clusters=self.b)
        self.features=features # training data for building the tree # dim (N_concatinate x 128)
        self.trainingImages=0   # count of the total images , used in tf-idf weighting
        self.database_index={}  # placeholder for the tf-idf score for each image eg : img_name : {leaf1:score1,leaf2:score2.......}
        self.database= None

    def fit(self,node,featuresIDs,depth):
        ''' Generates the vocabulary tree using hierarchical k-means clustering. Populates the node and tree dictionary.
        args:
            featureIDs (numpy.ndarray): index of the descriptors
            node (int): current node 
            depth (int): current depth from root (root=0)
        '''
        self.tree[node] = []
        if len(featuresIDs) >= self.b and depth < self.depth :
            self.model.fit([self.features[i] for i in featuresIDs])
            childIDs = [[] for i in range(self.b)]
            for i in range(len(featuresIDs)):
                childIDs[self.model.labels_[i]].append(featuresIDs[i])
            for i in range(self.b):
                self.currentnode = self.currentnode + 1
                self.nodes[self.currentnode] = self.model.cluster_centers_[i]
                self.tree[node].append(self.currentnode)
                self.fit(self.currentnode, childIDs[i], depth + 1)
        else:
            self.imageFreqInLeaves[node] = {}
    
    def find_best_leafID(self,descriptor, node):
        '''
        Function goes down the tree upto the leaf node to find the closest cluster centroid for given descriptor.
        args:
            descriptor: the descriptor to lookup
            node: the current node

        '''
        D_min = np.infty
        nextNode = None
        for child in self.tree[node]:
            dist = np.linalg.norm([self.nodes[child] - descriptor])
            if D_min > dist:
                D_min = dist
                nextNode = child
        if self.tree[nextNode] == []:
            return nextNode
        return self.find_best_leafID(descriptor, nextNode)

    def tfidf(self,filename,features):
        '''
        this method counts the image feature  frequency for each leaf node.
        args:
            filename :  eg: obj10
            features :  expects list of the form  [(obj10_sampl1,kp1),(obj10_sampl2,kp2),(obj10_sampl3,kp3)]}
        '''
        for i in range(len(features)):
            des=features[i][1]
            for d in des:
                leafID = self.find_best_leafID(d, 0)
                if filename in self.imageFreqInLeaves[leafID]:
                    self.imageFreqInLeaves[leafID][filename] += 1
                else:
                    self.imageFreqInLeaves[leafID][filename] = 1

    def build_tree(self):
        '''
        driver function for the fit method
        '''
        starting_node=0
        starting_depth=0
        featuresIDs = [x for x in range(len(self.features))]
        self.fit(starting_node, featuresIDs, starting_depth)

    def weight(self,leafID):
        '''
        calcualtes the weights for tf-idf weighting
        '''
        N=self.trainingImages
        return math.log1p(N/1.0*len(self.imageFreqInLeaves[leafID]))
    
    def generate_database_index(self,database):
        '''
        args:

            database : a dictionary in the form d[obj1]=[(sampl1,des1),(sampl2,des2),(sampl3,des3)]}
        '''
        self.database=database
        for k in database.keys():
            self.trainingImages+=len(database[k]) # finds out how many images in the dataset

        for key in database.keys():
            object_name=key
            self.tfidf(object_name,database[object_name])   # propogates each image down the tree . Each object_name has multiple images

        for leafID in self.imageFreqInLeaves:
            for img in self.imageFreqInLeaves[leafID]:      
                if img not in self.database_index:
                    self.database_index[img] = {}   #create dictionary for the first fill
                self.database_index[img][leafID] = self.weight(leafID)*(self.imageFreqInLeaves[leafID][img])    # multiply the counts by weight
        

        for img in self.database_index:
            denom = 0.0                     #normailze by the total counts 
            for leafID in self.database_index[img]:
                denom += self.database_index[img][leafID]
            for leafID in self.database_index[img]:
                self.database_index[img][leafID] /= denom
        



    def evaluate_querry(self,des):
        '''
        evalautes the querry score, same tf-idf weighting . returns the score 
        of each training image

        args: 
            des : descriptor of test image

        '''
        querry_index = {}
        for d in des:
            leafID = self.find_best_leafID(d, 0)
            if leafID in querry_index:
                querry_index[leafID] += 1
            else:
                querry_index[leafID] = 1
        denom = 0.0
        for key in querry_index:
            querry_index[key] = querry_index[key]*self.weight(key)
            denom += querry_index[key]
        for key in querry_index:
            querry_index[key] = querry_index[key]/denom
        
        scores = {}
        for img in self.database.keys():
            embedding_size=len(self.tree.keys())
            querry_embedding=np.zeros(embedding_size)   #make them the same dimension
            target_embedding=np.zeros(embedding_size)
            for k in querry_index.keys():
                querry_embedding[k]=querry_index[k]
            for k in self.database_index[img].keys():
                target_embedding[k]=self.database_index[img][k]
            scores[img]=np.linalg.norm((querry_embedding - target_embedding), ord=1)  #l1 norm used for score evaluation

            # scores[img]=np.linalg.norm((querry_embedding - target_embedding), ord=2)  #l2 norm used for score evaluation

            # scores[img] = spatial.distance.cosine(querry_embedding, target_embedding) #spatial.distance.cosine computes the distance, for sim subtract from 1
            
            scores={k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}  #sort dictionary on score values
        return scores
