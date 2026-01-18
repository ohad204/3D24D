import matplotlib.pyplot as plt

our_lpips = [
    0.127132177,
    0.061323762,
    0.19130826,
    0.178012848,
    0.06618762,
    0.084452748,
    0.24929595,
    0.276352167,
    0.119582176,
    0.178698778,
    0.12855494
]

our_dynamic = [
    0.565256102, 0.477573974, 0.758825583, 0.789950175, 0.523340417,
    0.550021572, 0.739487572, 0.580769761, 0.500148995, 0.618236585,
    0.527893914
]

our_style = [
    0.238037109,
    0.171508789,
    0.224853516,
    0.252258301,
    0.184997559,
    0.222808838,
    0.215820313,
    0.120010376,
    0.15234375,
    0.179534912,
    0.19140625
]

animate_dynamic = [
    1.909362197,
    0.582763433,
    1.909091234,
    0.652911186,
    1.237264872,
    0.456185222,
    1.181604385,
    0.839597642,
    1.381299734,
    1.065497875,
    0.385104686
]

animate_lpips = [
    0.359392166,
    0.284958005,
    0.409193039,
    0.443167686,
    0.390379906,
    0.193633318,
    0.383855343,
    0.323755264,
    0.43614006,
    0.401146889,
    0.273219109
]

animate_style = [
    0.194976807,
    0.289489746,
    0.33416748,
    0.195343018,
    0.236907959,
    0.217285156,
    0.277099609,
    0.205291748,
    0.246154785,
    0.249420166,
    0.218292236
]

dfy_style = [
    0.214447021,
    0.1509552,
    0.20703125,
    0.268341064,
    0.289611816,
    0.158172607,
    0.235137939,
    0.25579834,
    0.206451416,
    0.213287354,
    0.23727417
]

dfy_dynamic = [
    1.153839231,
    0.695961714,
    0.834039986,
    0.831599951,
    0.457166225,
    0.519273639,
    0.766779065,
    0.723727643,
    0.663876534,
    1.05048573,
    0.803198576
]

dfy_lpips = [
    0.52599144,
    0.584446907,
    0.378283501,
    0.50754261,
    0.21094656,
    0.258887291,
    0.578533173,
    0.441884041,
    0.368764877,
    0.261487007,
    0.540884018
]


stag3d_dynamic = [0.2468439698219303, 0.2390790551900864, 0.92275378505389, 0.699022833506266, 0.234688800573349,
                  0.253146562973658, 0.6451535860697426, 0.953220816453298, 0.969545819362005, 0.35026955405871,
                  0.502606294552486]

stag3d_style = [0.209961524605751, 0.1356937199831009, 0.1772851690649986, 0.1855317264795303, 0.1372592210769653,
                0.2710610508918762, 0.2069708108901978, 0.1868191719055176, 0.113301232457161, 0.1795948505401611,
                0.14856171905994415]

stag3d_lpips = [0.9327260553836823, 1.0288692861795425, 1.0093650221824646, 1.4437589943408966, 1.2922109365463257,
                1.1259135007858276, 1.2982087135314941, 0.876735508441925, 1.0607233792543411, 1.1769104301929474,
                0.8292776346206665]

# plt.scatter(our_lpips, our_dynamic); plt.xlabel('LPIPS'); plt.ylabel('Dynamic Range'); plt.show()
# plt.scatter(our_lpips, our_style); plt.xlabel('LPIPS'); plt.ylabel('Style'); plt.show()

plt.scatter(our_lpips, our_dynamic, color='blue', label='Ours')
plt.scatter(animate_lpips, animate_dynamic, color='red', label='Animate124')
plt.scatter(dfy_lpips, dfy_dynamic, color='green', label='4DFY')
plt.scatter(stag3d_lpips, stag3d_dynamic, color='orange', label='STAG3D')
plt.xlabel('LPIPS', fontsize=20)
plt.ylabel('Dynamic Degree', fontsize=20)
# plt.axis('square')
plt.legend(fontsize=15)
plt.title('LPIPS vs Dynamic Degree', fontsize=24)
plt.xticks(fontsize=11)  # Increase x-axis tick label font size
plt.yticks(fontsize=11)  # Increase y-axis tick label font size
plt.show()


plt.scatter(our_lpips, our_style, color='blue', label='Ours')
plt.scatter(animate_lpips, animate_style, color='red', label='Animate124')
plt.scatter(dfy_lpips, dfy_style, color='green', label='4DFY')
plt.scatter(stag3d_lpips, stag3d_style, color='orange', label='STAG3D')
plt.xlabel('LPIPS', fontsize=20)
plt.ylabel('temporal style', fontsize=20)
# plt.axis('square')
plt.legend(fontsize=15)
plt.title('LPIPS vs Temporal Style', fontsize=24)
plt.xticks(fontsize=11)  # Increase x-axis tick label font size
plt.yticks(fontsize=11)  # Increase y-axis tick label font size
plt.show()