# Install conda and rdkit
# import sys
# sys.path.append('/usr/local/lib/python3.7/site-packages/')

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import seaborn as sns
import matplotlib.pyplot as plt

# Reading csv file
df = pd.read_csv('bioactivity_data_curated.csv')
# print(df)


# Purpose of the bellow Code:
# The code cleans the canonical_smiles column by splitting each SMILES string into components using a period (.) as the separator and retaining only the longest component. This is useful when working with chemical data where SMILES strings may represent disconnected molecular structures, and you only want the primary (longest) component for analysis. The cleaned data is then recombined with the original DataFrame, replacing the original canonical_smiles column.
df_no_smiles = df.drop(columns='canonical_smiles')

smiles = []
for i in df.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)

smiles = pd.Series(smiles, name = 'canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles,smiles], axis=1)
# print(df_clean_smiles)

# Calculate Lipinski descriptors
def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)    # Converts each SMILES string into a molecular object using the RDKit library.
        moldata.append(mol)
       
    baseData= np.arange(1,1)    # Creates an empty NumPy array because the start and stop values are the same (1). 
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)                 # Computes the molecular weight.
        desc_MolLogP = Descriptors.MolLogP(mol)             # Computes the octanol-water partition coefficient (LogP).
        desc_NumHDonors = Lipinski.NumHDonors(mol)          # Counts the number of hydrogen bond donors.
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)    # Counts the number of hydrogen bond acceptors.
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
# print(df_lipinski)


# Now, let's combine the 2 DataFrame
df_combined = pd.concat([df,df_lipinski], axis=1)
# print(df_combined)

# Convert IC50 to pIC50
def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', axis=1)
        
    return x

# print(df_combined.standard_value.describe())
# print(-np.log10( (10**-9)* 100000000 ))
# print(-np.log10( (10**-9)* 10000000000 ))

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', axis=1)
        
    return x

df_norm = norm_value(df_combined)
# print(df_norm)

# print(df_norm.standard_value_norm.describe())

df_final = pIC50(df_norm)
# print(df_final)

# print(df_final.pIC50.describe())


# write this to CSV file.
df_final.to_csv('bioactivity_data_pIC50.csv')

# Removing the 'intermediate' bioactivity class
df_2class = df_final[df_final['class'] != 'intermediate']
# print(df_2class)

# Let's write this to CSV file.
df_2class.to_csv('bioactivity_data_2class_pIC50.csv')

# Exploratory Data Analysis (Chemical Space Analysis) via Lipinski descriptors
sns.set_theme(style='ticks')

# Frequency plot of the 2 bioactivity classes
plt.figure(figsize=(6.5, 6.5))                                  #  Initializes a new figure in matplotlib
sns.countplot(x='class', data=df_2class, edgecolor='black')     # A function from the seaborn library that creates a bar plot showing the frequency of unique values in a categorical variable.
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold') #  Set the labels for the x-axis
plt.ylabel('Frequency', fontsize=14, fontweight='bold')         #  Set the labels for the y-axis
plt.savefig('plot_bioactivity_class.pdf')                       # Saves the current figure to a file

# Scatter plot of MW versus LogP
plt.figure(figsize=(17.5, 17.5))
sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='class', size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

plt.savefig('plot_MW_vs_LogP.pdf')


# Box plots pIC50 value
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'class', y = 'pIC50', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')

plt.savefig('plot_ic50.pdf')



# Statistical analysis | Mann-Whitney U Test
def mannwhitney(descriptor, verbose=False):
  # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# actives and inactives
  selection = [descriptor, 'class']
  df = df_2class[selection]
  active = df[df['class'] == 'active']
  active = active[descriptor]

  selection = [descriptor, 'class']
  df = df_2class[selection]
  inactive = df[df['class'] == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'
  
  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv(filename)

  return results

print(mannwhitney('pIC50'))


#MW
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'class', y = 'MW', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('plot_MW.pdf')


print(mannwhitney('MW'))


# LogP
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'class', y = 'LogP', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.savefig('plot_LogP.pdf')

print(mannwhitney('LogP'))

# NumHDonors
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'class', y = 'NumHDonors', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')
plt.savefig('plot_NumHDonors.pdf')


print(mannwhitney('NumHDonors'))


# NumHAcceptors
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'class', y = 'NumHAcceptors', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')
plt.savefig('plot_NumHAcceptors.pdf')


print(mannwhitney('NumHAcceptors'))



# Zip files
# ! zip -r results.zip . -i *.csv *.pdf