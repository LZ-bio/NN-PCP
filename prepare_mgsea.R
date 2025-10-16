#1. To facilitate training, some operations in IGSEA are executed in the preprocessing stage.
rm(list = ls())
gc()
library(igraph)
library(foreach)
library(doParallel)
library(tsne)
library(lattice)
library(splines)
library(survival)
library(cluster)
library(survminer)
library(NMF)
library(kernlab)
library(snowfall)
library(parallel)
library(readr)
library(PMA)

rw=function(W,Fo,gamma) {
  ## perform a random walk; 
  maxIteration = 1000
  tof = 1e-6
  itcnt = 0
  Ft = Fo
  Ft1 = Fo
  while(itcnt < maxIteration)
  {
    Ft = (1-gamma)*(Ft %*% W) + gamma*(Fo)
    residual = max(apply(abs(Ft-Ft1),1,sum))
    Ft1 = Ft
    if(residual<tof)
    {
      print(paste('Iteration times:',itcnt))
      break
    }
    itcnt = itcnt + 1
  }
  return (Ft)
}

mnes = function(i){
  genescore = norMutationScoreNorm[i,]
  scorelocation = order(genescore,decreasing = T)
  genesort = networkgenes[scorelocation]
  pathwaynes = rep(0,FilesNum)
  for(j in 1:FilesNum)
  {
    pathwaygenes = lsRecord[j]
    pathwaygenes = pathwaygenes[[1]]
    genediff = setdiff(networkgenes,pathwaygenes)
    markscore = rep(0,length(networkgenes))
    loc = which(genesort %in% genediff)
    markscore[loc] = -1/length(loc)
    loc = which(genesort %in% pathwaygenes)
    markscore[loc] = 1/length(loc)
    markscoresum = cumsum(markscore)
    pathwaynes[j] = max(markscoresum)
  }
  return (pathwaynes)
}

cumfun1 <- function(x)
{
  y=1/10*x
  return (y)
}

cumfun2 <- function(x)
{
  y = 1/1000*x^3
  return (y)
}

cumfun3 <- function(x)
{
  y = 2/(1+exp(-x))-1
}

minmax <- function(x)
{
  y=(x-min(x))/(max(x)-min(x))
  return (y)
}


#############################
#1.1 handling SNV
Work.file = "G:/PhD/Mut_PR"
setwd(Work.file)
#####################parameter seting###############
cancers = c('PPRAD') 
Types = c('mut')
rwrParameter = 0.3
cannum = length(cancers)
for(h in 1:cannum)
{
  type = cancers[h]
  for(h1 in 1:2)
  {
    datatype = Types[h1]
    ##################section 1:input data##############
    #somatic mutation  
    file = paste('./data/cc/',type,'_',datatype,'_matrix.csv',sep='')
    data = read.csv(file,header = T,stringsAsFactors=FALSE,row.names = 1)
    cancerBarcode = rownames(data)
    cancerBarcodeLen = length(cancerBarcode)
    #PPI network
    network = read.delim("./data/networks/HumanNet_0.1_1.txt",header = TRUE, stringsAsFactors=FALSE)
    edge1=network[,1]
    edge2=network[,2]
    relationship=data.frame(edge1,edge2)
    point_0=data.frame(name=unique(union(edge1,edge2)),size=0)
    rownames(point_0)=point_0[,1]
    networkgenes = point_0[,1]
    g1=graph.data.frame(relationship,directed = FALSE,vertices = point_0)
    #adjweight=get.adjacency(g1,attr = 'weight',sparse = T)
    adjweight=get.adjacency(g1,sparse = T)
    matrixGeneNum = length(point_0[,1])
    adjweight = adjweight/apply(adjweight,1,sum)
    nor.adjweight = adjweight
    tum.adjweight = adjweight
    tum.networkgenes = nor.networkgenes = networkgenes
    #biological pathway(kegg+Pathwaycommons)
    file = './data/pathways/PK/PathwayUsed.txt'
    PathwayInfo = read.delim(file,header=F,sep='\t')
    pathways = unique(PathwayInfo[,2])
    FilesNum = length(pathways)
    PathwaysLens = rep(0,FilesNum)
    for(k1 in 1:FilesNum)
    {
      pathway = pathways[k1]
      loc = which(PathwayInfo[,2] %in% pathway)
      GeneName = PathwayInfo[loc,1]
      loc = match(GeneName,networkgenes)
      loc1 = which(!is.na(loc))
      temp = GeneName[loc1]
      PathwaysLens[k1] = length(temp)
      if(k1==1)
      {
        ls = list(temp)
      }else{
        ls = c(ls,list(temp))
      }
    }
    lsRecord = ls
    ######
    savefile = './data/result2/'
    if(!dir.exists(savefile))
    {
      dir.create(savefile)
    }
    ####
    networkgeneslen = length(networkgenes)
    MutationMatrix = matrix(0,cancerBarcodeLen,networkgeneslen)
    loc = match(networkgenes,colnames(data))
    loc1 = which(!is.na(loc))
    loc = as.numeric(na.omit(loc))
    for(i in 1:length(loc1))
    {
      MutationMatrix[,loc1[i]]=data[,loc[i]]
    }
    rowGeneNum <- apply(MutationMatrix,1,sum)
    loc1 = which(rowGeneNum==0)
    loc2 = which(is.na(rowGeneNum))
    loc1 = union(loc1,loc2)
    if(length(loc1)>0)
    {
      cancerBarcodeLen = cancerBarcodeLen - length(loc1)
      MutationMatrix=MutationMatrix[-loc1,]
      cancerBarcode = cancerBarcode[-loc1]
      rowGeneNum <-rowGeneNum[-loc1]
    }
    MutationScore = rw(adjweight,MutationMatrix,rwrParameter)
    MutationScoreNorm = MutationScore/rowGeneNum
    MutationScoreNorm = as.matrix(MutationScoreNorm)
    norMutationScoreNorm = MutationScoreNorm
    ###
    sfInit(parallel = TRUE,cpus = detectCores(logical = F))
    sfExport("lsRecord", local=FALSE)
    sfExport("norMutationScoreNorm", local=FALSE)
    sfExport("networkgenes", local=FALSE)
    sfExport("FilesNum", local=FALSE)
    result = sfLapply(1:cancerBarcodeLen, mnes)
    sfStop()
    Result = matrix(unlist(result),ncol=FilesNum,byrow=T)
    ##
    rownames(Result) <- cancerBarcode
    colnames(Result) <- pathways
    file = paste(savefile,'/',type,'_',datatype,'_Pathway_NES.csv',sep='')
    write.csv(Result,file, row.names = T,quote = T)
    
    mnessort = t(apply(Result,1,rank))
    unitvalue = FilesNum/20
    mnessort = mnessort/unitvalue - 10
    mmnes = cumfun3(mnessort)
    rownames(mmnes) <- cancerBarcode
    colnames(mmnes) <- pathways
    file = paste(savefile,'/',type,'_',datatype,'_mmnes.csv',sep='')
    write.csv(mmnes,file, row.names = T,quote = T)
  }
}

#1.2 handling CNV
Work.file = "G:/PhD/Mut_PR"
setwd(Work.file)
#####################parameter seting###############
cancers = c('PPRAD')
cannum = length(cancers)
Types = c('cnv')
cnvnames = c('cnv_del','cnv_amp','cnv') 
rwrParameter = 0.3

#PPI network
network = read.delim("./data/networks/HumanNet_0.1_1.txt",header = TRUE, stringsAsFactors=FALSE)
edge1=network[,1]
edge2=network[,2]
relationship=data.frame(edge1,edge2)
point_0=data.frame(name=unique(union(edge1,edge2)),size=0)
rownames(point_0)=point_0[,1]
networkgenes = point_0[,1]
networkgeneslen = length(networkgenes)
g1=graph.data.frame(relationship,directed = FALSE,vertices = point_0)
#adjweight=get.adjacency(g1,attr = 'weight',sparse = T)
adjweight=get.adjacency(g1,sparse = T)
matrixGeneNum = length(point_0[,1])
adjweight = adjweight/apply(adjweight,1,sum)
nor.adjweight = adjweight
tum.adjweight = adjweight
tum.networkgenes = nor.networkgenes = networkgenes
#biological pathway(kegg+Pathwaycommons)
file = './data/pathways/PK/PathwayUsed.txt'
PathwayInfo = read.delim(file,header=F,sep='\t')
pathways = unique(PathwayInfo[,2])
FilesNum = length(pathways)
PathwaysLens = rep(0,FilesNum)
for(k1 in 1:FilesNum)
{
  pathway = pathways[k1]
  loc = which(PathwayInfo[,2] %in% pathway)
  GeneName = PathwayInfo[loc,1]
  loc = match(GeneName,networkgenes)
  loc1 = which(!is.na(loc))
  temp = GeneName[loc1]
  PathwaysLens[k1] = length(temp)
  if(k1==1)
  {
    ls = list(temp)
  }else{
    ls = c(ls,list(temp))
  }
}
lsRecord = ls

######
savefile = './data/result2/'
if(!dir.exists(savefile))
{
  dir.create(savefile)
}

for(h in 1:cannum)
{
  type = cancers[h]
  for(h1 in 1:1)
  {
    datatype = Types[h1]
    ##################section 1:input data##############
    #somatic mutation 
    for(h2 in 1:3)
    {
      rname = cnvnames[h2]
      file = paste('./data/cc/',type,'_',datatype,'_matrix.csv',sep='')
      data = read.csv(file,header = T,stringsAsFactors=FALSE,row.names = 1)
      if(h2==1)
      {
        data[data>0] = 0
        data[data<0] = 1
      }else if(h2==2){
        data[data<0] = 0
        data[data>0] = 1
      }else{
        data[data!=0] = 1
      }
      cancerBarcode = rownames(data)
      cancerBarcodeLen = length(cancerBarcode)
      ####
      MutationMatrix = matrix(0,cancerBarcodeLen,networkgeneslen)
      loc = match(networkgenes,colnames(data))
      loc1 = which(!is.na(loc))
      loc = as.numeric(na.omit(loc))
      for(i in 1:length(loc1))
      {
        MutationMatrix[,loc1[i]]=data[,loc[i]]
      }
      rowGeneNum <- apply(MutationMatrix,1,sum)
      loc1 = which(rowGeneNum==0)
      loc2 = which(is.na(rowGeneNum))
      loc1 = union(loc1,loc2)
      if(length(loc1)>0)
      {
        cancerBarcodeLen = cancerBarcodeLen - length(loc1)
        MutationMatrix=MutationMatrix[-loc1,]
        cancerBarcode = cancerBarcode[-loc1]
        rowGeneNum <-rowGeneNum[-loc1]
      }
      MutationScore = rw(adjweight,MutationMatrix,rwrParameter)
      MutationScoreNorm = MutationScore/rowGeneNum
      MutationScoreNorm = as.matrix(MutationScoreNorm)
      norMutationScoreNorm = MutationScoreNorm
      ###
      sfInit(parallel = TRUE,cpus = detectCores(logical = F))
      sfExport("lsRecord", local=FALSE)
      sfExport("norMutationScoreNorm", local=FALSE)
      sfExport("networkgenes", local=FALSE)
      sfExport("FilesNum", local=FALSE)
      result = sfLapply(1:cancerBarcodeLen, mnes)
      sfStop()
      Result = matrix(unlist(result),ncol=FilesNum,byrow=T)
      ##
      rownames(Result) <- cancerBarcode
      colnames(Result) <- pathways
      file = paste(savefile,'/',type,'_',datatype,'_Pathway_NES.csv',sep='')
      write.csv(Result,file, row.names = T,quote = T)
      
      mnessort = t(apply(Result,1,rank))
      unitvalue = FilesNum/20
      mnessort = mnessort/unitvalue - 10
      mmnes = cumfun3(mnessort)
      rownames(mmnes) <- cancerBarcode
      colnames(mmnes) <- pathways
      file = paste(savefile,'/',type,'_',rname,'_mmnes.csv',sep='')
      write.csv(mmnes,file, row.names = T,quote = T)
    }
  }
}






