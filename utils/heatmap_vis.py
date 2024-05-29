import os 
import numpy as np

if True:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white") 
    sns.set(font_scale=5)
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False


# def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
#     def heatmap(data, camp='RdYlGn', figsize=(10, 10), ax=None, save_path=None):


data = np.random.randint(0, 255, (256, 256))/255.0

plt.figure(figsize=(10, 10), dpi=40)
ax = sns.heatmap(data,
                 xticklabels=False,
                 yticklabels=False, cmap='RdYlGn',
                 center=0, annot=False, ax=None, cbar=True, annot_kws={"size": 24}, fmt='.2f') 

ax.collections[0].set_clim(0,1) 
# ax.collections[0].colorbar.ax.tick_params(labelsize=20)
plt.savefig(r"E:\STUDY\Me\Publications\Submission\RWKV-IR\erf\cbar.png")