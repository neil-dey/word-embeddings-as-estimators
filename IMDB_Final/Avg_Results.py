import numpy as np
import pandas as pd

import os

base_path = sys.argv[1]

# list model names and k values then iterate over saving all resutls to a file
K = [1,2,5]
models = ['w2v_','SPPMI_','EM_matrix_','DD_matrix_','indep']
model_names = ['Word2Vec','SPPMI','EM_MVSVD','DD_MVSVD','Custom']
rows = []
for k in K:
    for model in range(len(models)):
        values = []
        for cv in range(20):
            try:
                preds = pd.read_csv(base_path + '/Results/'+models[model]+'_acc_k{}_cv{}.csv'.format(k,cv))
                acc = preds.iloc[1,0]
                values.append(acc)
            except:
                print('bad run')
        print(values)
        values = [x for x in values if x != 0]
        for obs in values:
            row = [model_names[model],k,obs]
            rows.append(row)

results = pd.DataFrame(rows,columns=['Models','k','obs'])
results.to_csv(base_path + '/results.csv',index=False)


import seaborn as sns
ax = sns.boxplot(x='Models',y='obs',hue='k',data=results)

ax.set(ylim=(.45, .9))
ax.set(title = 'Sentiment Analysis Accuracy')
ax.set(xlabel = 'Models')
ax.set(ylabel = 'Accuracy')

plt.legend(loc='lower right',title='k')
plt.savefig(base_path + '/LSTM_Results.png',dpi=400)

rows = []
for group in range(int(300/20)):
    groupData = results.iloc[group*20:(group+1)*20,:]
    obs = groupData['obs']
    avg = np.average(obs)
    std = np.std(obs)
    
    row = [groupData.iloc[0,0],groupData.iloc[0,1],avg,std]
    rows.append(row)
    
compiled_data = pd.DataFrame(rows,columns=['Model','k','Avg','Std'])

compiled_data.to_csv(base_path + '/results_compiled.csv',index=False)




















