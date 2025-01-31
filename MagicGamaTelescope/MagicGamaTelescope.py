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
    plt.hist(df[df['class']==0][label] , color='blue', label='hardon', alpha=0.7, density=True) # inside dataframe get me everything where class == 1  # density = true normalizes these distribution The density=True parameter in plt.hist() normalizes the histogram so that the total area under the histogram sums to 1. This is useful when comparing distributions with different sample sizes ( on y axis)
    plt.hist(df[df['class']==1][label] , color='hotpink', label='gamma', alpha=1, density=True) # inside dataframe get me everything where class == 1 
    plt.title(label)
    plt.ylabel('prob')
    plt.xlabel(label) #. A legend is a small box that explains what different colors, markers, or lines in the graph represent.
    plt.legend()
    plt.show()
 
#df.sample(frac=1) shuffles the dataset randomly.
#frac=1 means we take 100% of the data but in random order.
#This ensures that the split is randomized instead of sequential. 


   
train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int (0.8 * len(df))]) # The first 60% of shuffled data goes into train The next 20% (from 60% to 80%) goes into valid (validation set). The remaining 20% (from 80% to 100%) goes into test (testing set).

# scale these so that it will become realtive to the mean and standard deviation 

