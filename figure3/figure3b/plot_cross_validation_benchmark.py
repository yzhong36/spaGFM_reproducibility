
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})


def mean_std_sem(values):
    values = np.asarray(values)
    mean = values.mean()
    std = values.std(ddof=1)      # sample std
    sem = std / np.sqrt(len(values))
    return (mean, sem)



mcc= {
    'PCA':[0.0, 0.0, 0.0, 0.0],
    'NicheCompass':[0.056, 0.026, 0.085, 0.058],
    'scGPT':[0.232, 0.062, 0.259, 0.217],
    'finetune-scGPT':[0.228, 0.152, 0.227, 0.229],
    'NOVAE': [np.float64(0.08724973440132214), np.float64(0.11821496404617622), np.float64(0.08526032698262066), np.float64(0.10371173366530773)],
    'scGPT-spatial': [np.float64(-0.009747366907712933), np.float64(0.03081471449942908), np.float64(0.010879112290147378), np.float64(-0.010855267085412318)],
    'spaGFM':[0.634, 0.595, 0.644, 0.460],

}

# f1= {
#     'Scanpy':[0.317, 0.317, 0.317, 0.317],
#     'scGPT-spatial': [0.3193639266136772, 0.3321930424480614, 0.3266939727042777, 0.317055123936367],
#     'NicheCompass':[0.328, 0.317, 0.340, 0.328],
#     'NOVAE': [0.3617958179501584, 0.35647361921799653, 0.3596529617329916, 0.3710448299331073],
#     'scGPT':[0.421, 0.340, 0.428, 0.414],
#     'finetune-scGPT':[0.431, 0.396, 0.458, 0.447],
#     'spaGFM':[0.716, 0.6306, 0.668, 0.596],
#
# }

f1= {
    # 'Scanpy':[0.317, 0.317, 0.317, 0.317],
    'scGPT-spatial': [0.3256704980842912, 0.32702655788925533, 0.3193639266136772, 0.3416045966295614, 0.32931928416123096],
    'NOVAE': [0.3279745262503883, 0.3279672263573943, 0.3175747492952539, 0.3236155314088835, 0.375630799753159],
    'NicheCompass':[0.3468989555644058, 0.42727023603586023, 0.3281933120642798, 0.3193379613481537, 0.3570796829370046],
    'scGPT':[0.40631407899203426, 0.44336479903950227, 0.4209743610941412, 0.34494738066166636, 0.36787161974568267],
    'finetune-scGPT':[0.4030206753107975, 0.4751682050995643, 0.43095879782018925, 0.34614492753623187, 0.3606679993336665],
    'spaGFM':[0.6601472554058762, 0.5664005416294055, 0.5633032270735164, 0.42016802769423434, 0.4187821389051618],

}

recall = {
    'PCA':[0.333, 0.333, 0.333, 0.333],
    'NicheCompass': [0.338, 0.333, 0.345, 0.338],
    'scGPT':[0.406, 0.344, 0.431, 0.400],
    'finetune-scGPT':[0.446, 0.378, 0.462, 0.450],
    'NOVAE' : [0.3576316832852491, 0.35588230331865506, 0.3609257689257816, 0.36491718125221434],
    'scGPT-spatial' : [0.331924233604967, 0.3393839314335847, 0.33657596041909194, 0.3325572370974001],
    'spaGFM':[0.681, 0.579, 0.677, 0.623],
}
precision = {
    'PCA':[0.303, 0.303, 0.303, 0.303],
    'NicheCompass': [0.349, 0.303, 0.363, 0.347],
    'scGPT':[0.583, 0.415, 0.520, 0.558],
    'finetune-scGPT':[0.511, 0.514, 0.435, 0.470],
    'NOVAE' : [0.3982065101549892, 0.3927636674749786, 0.37032276276964926, 0.40263316790593],
    'scGPT-spatial': [0.31568885417094933, 0.36998040263673615, 0.3375535196353759, 0.30293389890420647],
    'spaGFM':[0.768, 0.719, 0.704, 0.575],
}


colors = {
    "Scanpy": "#D5EAD9",   # light purple
    "NicheCompass": "#7DC69B",   # blue
    "NOVAE":  "#9BD7F3",   # magenta
    "finetune-scGPT": "#FBDDDD",
    "scGPT-spatial": "#C9B6E4",
    "spaGFM":"#F2A1A7",
    "scGPT": "#FCE6CF"
}

metrics = {}

for methods, values in f1.items():
    metrics[methods] = mean_std_sem(values)

x = np.arange(len(metrics.keys()))
# -----------------------------
# Plot
# -----------------------------

for idx, (method, (mean, err)) in enumerate(metrics.items()):
    plt.bar(
        x[idx],
        mean,
        yerr=err,
        label=method,
        color=colors[method],
        capsize=3
    )

# plt.set_title(metric, fontsize=11)
plt.xticks(x, metrics.keys())
plt.ylim(0, 0.6)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(axis="both", labelsize=9)
# plt.legend()
plt.savefig(r'C:\Users\hef\Downloads\spaGFM\5fold_benchmark_f1.png', dpi=600, bbox_inches='tight')