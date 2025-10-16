import numpy as np
import pandas as pd
import networkx as nx
import math
import copy
from scipy.special import expit  # For sigmoid function
from tqdm import tqdm
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from scipy.stats import pearsonr


def random_walk(W, Fo, gamma, max_iterations=1000, tolerance=1e-6):
    Ft = Fo
    Ft1 = Fo
    for itcnt in range(max_iterations):
        Ft = (1 - gamma) * (Ft @ W) + gamma * Fo
        residual = np.max(np.abs(Ft - Ft1).sum(axis=1))
        Ft1 = Ft
        if residual < tolerance:
            print(f'Iteration times: {itcnt}')
            break
    return Ft

def pathway_nes(i, genescore, networkgenes, lsRecord):
    scorelocation = np.argsort(-genescore)
    genesort = networkgenes[scorelocation]
    pathwaynes = np.zeros(len(lsRecord))
    for j in range(len(lsRecord)):
        pathwaygenes = lsRecord[j]
        genediff = np.setdiff1d(networkgenes, pathwaygenes)
        markscore = np.zeros(len(networkgenes))
        loc = np.isin(genesort, genediff)
        markscore[loc] = -1 / np.sum(loc)
        loc = np.isin(genesort, pathwaygenes)
        markscore[loc] = 1 / np.sum(loc)
        markscoresum = np.cumsum(markscore)
        pathwaynes[j] = np.max(markscoresum)
    return pathwaynes

def nes(i,genesort,networkgenes,lsRecord):
    pathwaygenes = lsRecord[i]
    genediff = np.setdiff1d(networkgenes, pathwaygenes)
    markscore = np.zeros(len(networkgenes))
    loc = np.isin(genesort, genediff)
    markscore[loc] = -1 / np.sum(loc)
    loc = np.isin(genesort, pathwaygenes)
    markscore[loc] = 1 / np.sum(loc)
    markscoresum = np.cumsum(markscore)
    return np.max(markscoresum)

def pathway_nes1(nor_mutation_score_norm,networkgenes,lsRecord):
    len1 = nor_mutation_score_norm.shape[0]
    pathwaynes = []
    for i in range(len1):
        genescore = nor_mutation_score_norm[i]
        scorelocation = np.argsort(-genescore)
        genesort = networkgenes[scorelocation]
        ness = Parallel(n_jobs=-1)(delayed(nes)(i, genesort, networkgenes, lsRecord) for i in range(len(lsRecord)))
        pathwaynes.append(ness)


def cumfun3(x):
    return 2 / (1 + np.exp(-x)) - 1

def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))



#1.construct network
def mgsea(x,y,PathwayGenes,chosedPathways):
    rwr_parameter = 0.3
    permutation = 1000
    # Load PPI network
    network = pd.read_csv("./data/networks/HumanNet_0.1_1.txt",sep='\t',names=['EntrezGeneID','EntrezGeneID2'],header=None)
    # Create graph and adjacency matrix
    G = nx.from_pandas_edgelist(network, 'EntrezGeneID', 'EntrezGeneID2')
    nodes = list(G.nodes)
    adj_weight = nx.adjacency_matrix(G, nodelist=nodes)
    adj_weight /= adj_weight.sum(axis=1)
    # Biological pathways
    pathway_info = PathwayGenes[(PathwayGenes['gene'].isin(nodes)) & (PathwayGenes['group'].isin(chosedPathways))]
    print(pathway_info)
    pathways = pathway_info['group'].unique()
    pathway_nums = len(pathways)
    # Prepare lsRecord
    ls_record = []
    for pathway in pathways:
        genes = pathway_info[pathway_info['group'] == pathway]['gene'].values
        ls_record.append(genes)
    #load cancer data
    cancer_barcode = list(x.index)
    filter_matrix = pd.DataFrame(index=cancer_barcode,columns=nodes)
    filter_matrix.update(x)
    print(filter_matrix)
    print(x.columns)
    filter_matrix = filter_matrix.fillna(0)
    mutation_matrix = filter_matrix.values

    row_gene_num = mutation_matrix.sum(axis=1,keepdims=True)
    mutation_score = random_walk(adj_weight, mutation_matrix, rwr_parameter)
    mutation_score_norm = mutation_score / row_gene_num
    nor_mutation_score_norm = np.array(mutation_score_norm)

    # Calculate pathway NES
    results = Parallel(n_jobs=-1)(delayed(pathway_nes)(i, nor_mutation_score_norm[i], np.array(nodes).astype(str), ls_record) for i in range(len(cancer_barcode)))
    #results = pathway_nes1(nor_mutation_score_norm,np.array(nodes).astype(str),ls_record)
    result = np.array(results)
    sorted_indices = np.argsort(result, axis=1)
    positions = np.zeros_like(sorted_indices)
    # 填充每个元素的原始位置
    for i in range(result.shape[0]):
        positions[i, sorted_indices[i]] = np.arange(result.shape[1]) + 1

    unitvalue = pathway_nums / 20
    mnessort = positions / unitvalue - 10
    mmnes = cumfun3(mnessort)
    y_label = copy.deepcopy(y)
    
    for h3 in range(1, permutation + 1):
        if h3 == 1:
            cla = sorted(np.unique(y_label))
        loc = y_label == cla[0]
        normmnes = mmnes[loc]
        loc = y_label == cla[1]
        tummmnes = mmnes[loc]
        mmnes1 = np.sum(normmnes, axis=0)
        normatrix = mmnes1 if h3 == 1 else np.vstack([normatrix, mmnes1])
        mmnes1 = np.sum(tummmnes, axis=0)
        tummatrix = mmnes1 if h3 == 1 else np.vstack([tummatrix, mmnes1])
        np.random.shuffle(y_label)

    mmnesrank = rankdata(normatrix, axis=0)
    norpvalue = (permutation - mmnesrank[0]) / permutation
    mmnessum1 = normatrix[0]

    mmnesrank = rankdata(tummatrix, axis=0)
    tumpvalue = (permutation - mmnesrank[0]) / permutation
    mmnessum2 = tummatrix[0]

    norpvalue_adjusted = multipletests(norpvalue, method='bonferroni')[1]
    tumqvalue_adjusted = multipletests(tumpvalue, method='bonferroni')[1]
    norpvalue_fdr = multipletests(norpvalue, method='fdr_bh')[1]
    tumqvalue_fdr = multipletests(tumpvalue, method='fdr_bh')[1]

    result = pd.DataFrame({
        'mmnessum1': mmnessum1,
        'mmnessum2': mmnessum2,
        'nor_pvalue': norpvalue,
        'nor_bonferroni': norpvalue_adjusted,
        'nor_fdr': norpvalue_fdr,
        'tum_pvalue': tumpvalue,
        'tum_bonferroni': tumqvalue_adjusted,
        'tum_fdr': tumqvalue_fdr
    }, index=pathways)

    return result


#1.construct network
def cal_nes(x,PathwayGenes,name,cancer):
    rwr_parameter = 0.3
    # Load PPI network
    network = pd.read_csv("./data/networks/HumanNet_0.1_1.txt",sep='\t',names=['EntrezGeneID','EntrezGeneID2'],header=None)
    # Create graph and adjacency matrix
    G = nx.from_pandas_edgelist(network, 'EntrezGeneID', 'EntrezGeneID2')
    nodes = list(G.nodes)
    adj_weight = nx.adjacency_matrix(G, nodelist=nodes)
    adj_weight /= adj_weight.sum(axis=1)
    # Biological pathways
    pathway_info = PathwayGenes[PathwayGenes['gene'].isin(nodes)]
    print(pathway_info)
    pathways = pathway_info['group'].unique()
    pathway_nums = len(pathways)
    # Prepare lsRecord
    ls_record = []
    for pathway in pathways:
        genes = pathway_info[pathway_info['group'] == pathway]['gene'].values
        ls_record.append(genes)
    #load cancer data
    cancer_barcode = list(x.index)
    filter_matrix = pd.DataFrame(index=cancer_barcode,columns=nodes)
    filter_matrix.update(x)
    filter_matrix = filter_matrix.fillna(0)
    mutation_matrix = filter_matrix.values

    row_gene_num = mutation_matrix.sum(axis=1)
    mutation_matrix = mutation_matrix[row_gene_num!=0]
    cancer_barcode = np.array(cancer_barcode)
    cancer_barcode = list(cancer_barcode[row_gene_num!=0])
    row_gene_num = mutation_matrix.sum(axis=1,keepdims=True)
    mutation_score = random_walk(adj_weight, mutation_matrix, rwr_parameter)
    mutation_score_norm = mutation_score / row_gene_num
    nor_mutation_score_norm = np.array(mutation_score_norm)

    # Calculate pathway NES
    results = Parallel(n_jobs=-1)(delayed(pathway_nes)(i, nor_mutation_score_norm[i], np.array(nodes).astype(str), ls_record) for i in range(len(cancer_barcode)))
    #results = pathway_nes1(nor_mutation_score_norm,np.array(nodes).astype(str),ls_record)
    result = np.array(results)
    sorted_indices = np.argsort(result, axis=1)
    positions = np.zeros_like(sorted_indices)
    # 填充每个元素的原始位置
    for i in range(result.shape[0]):
        positions[i, sorted_indices[i]] = np.arange(result.shape[1]) + 1

    unitvalue = pathway_nums / 20
    mnessort = positions / unitvalue - 10
    mmnes = cumfun3(mnessort)
    mmnes = pd.DataFrame(mmnes,index=cancer_barcode,columns=pathways)
    file = "/home/zhouli/PhD/Mut_PR/data/bb/" + cancer + '_' + name + '_nes.csv'
    mmnes.to_csv(file, index=False)


def permutation_nes(i, cla, y, mmnes):
    y_label = copy.deepcopy(y)
    if i>0:
        np.random.shuffle(y_label)
    loc = y_label == cla[0]
    normmnes = mmnes[loc]
    loc = y_label == cla[1]
    tummmnes = mmnes[loc]
    mmnes1 = np.sum(normmnes, axis=0)
    mmnes2 = np.sum(tummmnes, axis=0)
    result = np.concatenate((mmnes1, mmnes2), axis=0)
    return result


def cal_mgsea(response,cancer,dataname,chosedPathways):
    permutation = 1000
    file = "/home/qmliu/zhouli/Mut_PR_1/data/result2/" + cancer + '_' + dataname + '_mmnes.csv'
    mmnes = pd.read_csv(file,index_col = 0)
    cancer_barcode = list(response.index)

    filter_matrix = pd.DataFrame(index=cancer_barcode,columns=chosedPathways)
    filter_matrix.update(mmnes)
    filter_matrix = filter_matrix.fillna(0)
    zero_rows = filter_matrix.eq(0).all(axis=1)
    filter_matrix = filter_matrix[zero_rows==False]
    mmnes = filter_matrix.values

    y1 = response['response'].values
    y1 = y1.reshape(-1)
    cla = sorted(np.unique(y1))
    y1 = y1[zero_rows==False]

    results = Parallel(n_jobs=-1)(delayed(permutation_nes)(i, cla, y1, mmnes) for i in range(permutation + 1))
    result = np.array(results)
    normatrix = result[:,0:len(chosedPathways)]
    tummatrix = result[:,len(chosedPathways):(2*len(chosedPathways))]

    mmnesrank = rankdata(normatrix, axis=0)
    norpvalue = (permutation - mmnesrank[0]) / permutation
    mmnessum1 = normatrix[0]

    mmnesrank = rankdata(tummatrix, axis=0)
    tumpvalue = (permutation - mmnesrank[0]) / permutation
    mmnessum2 = tummatrix[0]

    norpvalue_adjusted = multipletests(norpvalue, method='bonferroni')[1]
    tumqvalue_adjusted = multipletests(tumpvalue, method='bonferroni')[1]
    norpvalue_fdr = multipletests(norpvalue, method='fdr_bh')[1]
    tumqvalue_fdr = multipletests(tumpvalue, method='fdr_bh')[1]

    result = pd.DataFrame({
        'mmnessum1': mmnessum1,
        'mmnessum2': mmnessum2,
        'nor_pvalue': norpvalue,
        'nor_bonferroni': norpvalue_adjusted,
        'nor_fdr': norpvalue_fdr,
        'tum_pvalue': tumpvalue,
        'tum_bonferroni': tumqvalue_adjusted,
        'tum_fdr': tumqvalue_fdr
    }, index=chosedPathways)

    return result





#calculate person correlation coffe
def calculate_correlation_matrix(X):
    correlation_matrix = np.zeros((X.shape[0],X.shape[0]), dtype=np.float)
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            r, p = pearsonr(X[i, :], X[j, :])
            correlation_matrix[i, j] = correlation_matrix[j, i] = r
    return correlation_matrix

#imputing samples
def imputation(x,Barcodes,KNN_matrix):
    x1 = x.values
    result = copy.deepcopy(x)
    mutation_num = x1.sum(axis=1)
    mutation_zero = np.where(mutation_num==0)[0]
    if len(mutation_zero)>0:
        samples = x.index
        for h in mutation_zero:
            bar = samples[h]
            loc = list(Barcodes).index(bar)
            loc1 =  KNN_matrix[loc,1:]
            bars = Barcodes[loc1]
            x2 = x[samples.isin(bars)]
            temp = np.array(x2.sum(axis=0))
            temp[temp>0] = 1
            result.iloc[h,:] = temp
    else:
        return x
    return result

#construction network
def createNetwork1(pheName,dataname,chosedPathways,norPathways,tumPathways,numNeu):
    norneunames = [dataname + '_' + pheName[0] + '_' + str(h) for h in range(numNeu)] 
    tumneunames = [dataname + '_' + pheName[1] + '_' + str(h) for h in range(numNeu)]
    samneunames = [dataname + '_' + pheName[2] + '_' + str(h) for h in range(numNeu)]
    nornetwork = [[h,k] for h in norPathways for k in norneunames]
    tumnetwork = [[h,k] for h in tumPathways for k in tumneunames]
    samnetwork = [[h,k] for h in chosedPathways for k in samneunames]
    nornetwork = pd.DataFrame(nornetwork, columns=['Pathway', 'Pheno'])
    tumnetwork = pd.DataFrame(tumnetwork, columns=['Pathway', 'Pheno'])
    samnetwork = pd.DataFrame(samnetwork, columns=['Pathway', 'Pheno'])
    norneunames.extend(tumneunames)
    norneunames.extend(samneunames)
    names = norneunames
    network = pd.concat([nornetwork,tumnetwork,samnetwork])
    net = nx.from_pandas_edgelist(network, 'Pathway', 'Pheno', create_using=nx.DiGraph())
    return net, names


def createNetwork2(mrnaName,ampName,delName,dmethName,numNeu):
    norneunames = ['nor_' + str(h) for h in range(numNeu[1])] 
    tumneunames = ['tum_' + str(h) for h in range(numNeu[1])]
    samneunames = ['sam_' + str(h) for h in range(numNeu[1])]
    Name = [mrnaName,ampName,delName,dmethName]
    networks = []
    nas = []
    for name in Name:
        nas.extend(name)
        for i in range(3):
            temp = name[i*numNeu[0]:(i+1)*numNeu[0]]
            if i==0:    
                network = [[h,k] for h in temp for k in norneunames]
            elif i==1:
                network = [[h,k] for h in temp for k in tumneunames]
            else:
                network = [[h,k] for h in temp for k in samneunames]
            networks.extend(network)
    networks = pd.DataFrame(networks, columns=['Pheno', 'Pheno1'])
    norneunames.extend(tumneunames)
    norneunames.extend(samneunames)
    names = norneunames
    net = nx.from_pandas_edgelist(networks, 'Pheno', 'Pheno1', create_using=nx.DiGraph())
    return net, nas,names


def createNetwork4(mrnaName,ampName,delName,numNeu):
    norneunames = ['nor_' + str(h) for h in range(numNeu[1])] 
    tumneunames = ['tum_' + str(h) for h in range(numNeu[1])]
    samneunames = ['sam_' + str(h) for h in range(numNeu[1])]
    Name = [mrnaName,ampName,delName]
    networks = []
    nas = []
    for name in Name:
        nas.extend(name)
        for i in range(3):
            temp = name[i*numNeu[0]:(i+1)*numNeu[0]]
            if i==0:    
                network = [[h,k] for h in temp for k in norneunames]
            elif i==1:
                network = [[h,k] for h in temp for k in tumneunames]
            else:
                network = [[h,k] for h in temp for k in samneunames]
            networks.extend(network)
    networks = pd.DataFrame(networks, columns=['Pheno', 'Pheno1'])
    norneunames.extend(tumneunames)
    norneunames.extend(samneunames)
    names = norneunames
    net = nx.from_pandas_edgelist(networks, 'Pheno', 'Pheno1', create_using=nx.DiGraph())
    return net, nas,names


def createNetwork5(mrnaName,ampName,numNeu):
    norneunames = ['nor_' + str(h) for h in range(numNeu[1])] 
    tumneunames = ['tum_' + str(h) for h in range(numNeu[1])]
    samneunames = ['sam_' + str(h) for h in range(numNeu[1])]
    Name = [mrnaName,ampName]
    networks = []
    nas = []
    for name in Name:
        nas.extend(name)
        for i in range(3):
            temp = name[i*numNeu[0]:(i+1)*numNeu[0]]
            if i==0:    
                network = [[h,k] for h in temp for k in norneunames]
            elif i==1:
                network = [[h,k] for h in temp for k in tumneunames]
            else:
                network = [[h,k] for h in temp for k in samneunames]
            networks.extend(network)
    networks = pd.DataFrame(networks, columns=['Pheno', 'Pheno1'])
    norneunames.extend(tumneunames)
    norneunames.extend(samneunames)
    names = norneunames
    net = nx.from_pandas_edgelist(networks, 'Pheno', 'Pheno1', create_using=nx.DiGraph())
    return net, nas,names



def createNetwork3(mrnaName,ampName,dmethName,numNeu):
    norneunames = ['nor_' + str(h) for h in range(numNeu[1])] 
    tumneunames = ['tum_' + str(h) for h in range(numNeu[1])]
    samneunames = ['sam_' + str(h) for h in range(numNeu[1])]
    Name = [mrnaName,ampName,dmethName]
    networks = []
    nas = []
    for name in Name:
        nas.extend(name)
        for i in range(3):
            temp = name[i*numNeu[0]:(i+1)*numNeu[0]]
            if i==0:    
                network = [[h,k] for h in temp for k in norneunames]
            elif i==1:
                network = [[h,k] for h in temp for k in tumneunames]
            else:
                network = [[h,k] for h in temp for k in samneunames]
            networks.extend(network)
    networks = pd.DataFrame(networks, columns=['Pheno', 'Pheno1'])
    norneunames.extend(tumneunames)
    norneunames.extend(samneunames)
    names = norneunames
    net = nx.from_pandas_edgelist(networks, 'Pheno', 'Pheno1', create_using=nx.DiGraph())

    return net, nas,names


import random
import math


def imputation2(x1,x2,x3,x4,x5,y,mn):
    cla = np.unique(y)
    samplenum = []
    for i in cla:
        cl = list(np.where(y==i)[0])
        cllen = len(cl)
        samplenum.append(cllen)
    samplenum = np.array(samplenum)
    loc1 = np.argmin(samplenum)
    loc2 = 1 - loc1
    BSS = round((samplenum[loc2] - samplenum[loc1])/2)
    cl = list(np.where(y==cla[loc1])[0])
    x11 = copy.deepcopy(x1)
    x12 = copy.deepcopy(x2)
    x13 = copy.deepcopy(x3)
    x14 = copy.deepcopy(x4)
    x15 = copy.deepcopy(x5)

    y1 = copy.deepcopy(y)
    y11 = np.repeat(cla[loc1],BSS)
    for j in range(BSS):
        #loc = random.sample(cl,1)[0]
        #t_data = tuple(x_batch[loc])
        loc = random.sample(cl,mn)
        t_data1 = np.mean(x1[loc],axis=0)
        t_data = tuple(t_data1)
        x11 = np.row_stack((x11,t_data))
        t_data1 = np.mean(x2[loc],axis=0)
        t_data = tuple(t_data1)
        x12 = np.row_stack((x12,t_data))
        t_data1 = np.mean(x3[loc],axis=0)
        t_data = tuple(t_data1)
        x13 = np.row_stack((x13,t_data))
        t_data1 = np.mean(x4[loc],axis=0)
        t_data = tuple(t_data1)
        x14 = np.row_stack((x14,t_data))
        t_data1 = np.mean(x5[loc]).reshape(-1)
        x15 = np.concatenate((x15,t_data1))
    y1 = np.concatenate([y1, y11])
    return x11,x12,x13,x14,x15,y1

