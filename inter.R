pathways = read.delim("/home/zhouli/PhD/Mut_PR/data/pathways/PK/PathwayNames.txt",header = FALSE, stringsAsFactors=FALSE)

muts = c('SNA','AMP','DEL')
#files = c('m10.csv','m11.csv','m12.csv')
files = c('abs_m10.csv','abs_m11.csv','abs_m12.csv')
cancer = 'PPRAD'
len1 = length(muts)
for(i in 1:len1)
{
  file = files[i]
  mut = muts[i]
  File = paste("/home/zhouli/PhD/Mut_PR/data/coef/",cancer,'/average/',file,sep='')
  data = read.csv(File,header = T,stringsAsFactors=FALSE)
  data1 = data[1:20,]
  loc = match(data1[,1],pathways[,3])
  name = pathways[loc,1]
  data2 = cbind(data1,name)
  colnames(data2) = c('ID','PI','Name')
  File = paste("/home/zhouli/PhD/Mut_PR/data/coef/",cancer,'/bio_',mut,'_.csv',sep='')
  write.csv(data2,File, quote = T)
}



