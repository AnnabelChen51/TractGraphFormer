import numpy
import seaborn as sns
import matplotlib.pyplot as plt

preds=[]
for i in range(5):
    file = 'runs/DGCNNAMG_Transformer_cv_hcp/DGCNNAMG_Transformer_004/preds_com_{}.npy'.format(i)
    pred=list(numpy.load(file))
    preds.extend(pred)
preds = numpy.array(preds)
numpy.save('preds_AM_hcp.npy', preds)

preds=[]
for i in range(5):
    file = 'runs/DGCNNAMG_Transformer_cv_hcp/DGCNNAMG_Transformer_004/scale_{}.npy'.format(i)
    pred=list(numpy.load(file).squeeze())
    preds.extend(pred)
preds = numpy.array(preds)
numpy.save('weights_AM_hcp.npy', preds)


preds=[]
for i in range(5):
    file = 'runs/DGCNNGAMG_Transformer_cv/DGCNNAMG_Transformer_00{}/preds_com.npy'.format(i+1)
    pred=list(numpy.load(file))
    preds.extend(pred)
preds = numpy.array(preds)
numpy.save('preds_AM_abcd.npy', preds)

preds=[]
for i in range(5):
    file = 'runs/DGCNNGAMG_Transformer_cv/DGCNNAMG_Transformer_00{}/scale.npy'.format(i+1)
    pred=list(numpy.load(file).squeeze())
    preds.extend(pred)
preds = numpy.array(preds)
numpy.save('weights_AM_abcd.npy', preds)

preds=[]
for i in range(5):
    #file = 'runs/DGCNNG_Transformer_cv/DGCNNG_Transformer_0{}/preds_com.npy'.format(str(i + 1).zfill(2))
    file = 'runs/DGCNNG_Transformer_cv_hcp_wb_best/DGCNNG_Transformer_001/preds_com_{}.npy'.format(str(i))
    pred=list(numpy.load(file))
    preds.extend(pred)
preds = numpy.array(preds)
numpy.save('preds_hcp.npy', preds)

weights=[]
for i in range(5):
    file='runs/DGCNNG_Transformer_cv/DGCNNG_Transformer_0{}/weights_{}.npy'.format(str(i+16).zfill(2),i)
    weight=list(numpy.load(file))
    weights.extend(weight)
weights=numpy.array(weights)
numpy.save('weights_abcd3.npy',weights)

weight1=numpy.load('weights_abcd3.npy')
weight2=numpy.load('weights_hcp.npy')

weight1n=(weight1-weight1.mean(1,keepdims=True))/weight1.std(1,keepdims=True)
weight2n=(weight2-weight2.mean(1,keepdims=True))/weight2.std(1,keepdims=True)

sns.heatmap(weight2n, cmap='viridis',  fmt=".2f", cbar_kws={'label': 'Scale'})
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap Example')
plt.show()
