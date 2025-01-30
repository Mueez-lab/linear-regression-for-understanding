# ----------------------------------------------------- Classification ----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols=('fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist', 'class')
df = pd.read_csv('C:/Users/mueez/OneDrive/Desktop/Machine/MagicGamaTelescope/magic04.data', names=cols)
# g is gama and h is hadrons
# computer is not good at understanding letters so we will convert g and h into number
df['class'] = (df['class'] == 'g').astype(int); # covert g into 0 or 1 and h into oposite of that
print(df.head()) # 1st five
for label in cols[:-1]: # from start till end 
    plt.hist(df[df['class']==1][label] , color='hotpink', label='gamma', alpha=0.7, density=True) # inside dataframe get me everything where class == 1  # density = true normalizes these distribution 
    plt.hist(df[df['class']==2][label] , color='hotpink', label='gamma', alpha=0.7, density=True) # inside dataframe get me everything where class == 1 
    plt.title(label)
    plt.ylabel('prob')
    plt.xlabel(label)
    plt.legend()
    plt.show()