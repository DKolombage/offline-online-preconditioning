import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import build_coefficient

#---Offline coefficients

def plot_offline_coeff(Nepsilon, NFine, NCoarse, k, alpha, beta, type):
    NCoarseElement = NFine // NCoarse
    world = wrld.World(NCoarse, NCoarseElement, None)
    patch = wrld.construct_center_node_patch(world, k)  
    xpFine = util.pCoordinates(world.NWorldCoarse, patch.iPatchWorldCoarse, patch.NPatchCoarse)
    if type=="checker_board":        
        aRefList_rand = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, Nepsilon // NCoarse, world.NCoarseElement, alpha, beta)
        fig = plt.figure()
        for ii in range(4):
            ax = fig.add_subplot(1, 4, ii+1)
            apertGrid = aRefList_rand[-1+ii].reshape(patch.NPatchFine, order='C')
            im = ax.imshow(apertGrid, origin='lower', extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greens')
        #plt.figure(figsize=(10,4))
        #plt.subplot_tool()
        fig.tight_layout()
        fig.savefig("offline.png")
        plt.show()

    if type=="erasure":
        incl_bl = np.array([0.25, 0.25])
        incl_tr = np.array([0.75, 0.75])
        aRefList_incl = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse, Nepsilon // NCoarse, world.NCoarseElement, alpha, beta, incl_bl, incl_tr)
        fig = plt.figure()
        for ii in range(4):
            ax = fig.add_subplot(2, 2, ii+1)
            apertGrid = aRefList_incl[-1+ii].reshape(patch.NPatchFine, order='C')
            im = ax.imshow(apertGrid, origin='lower',
                        extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap='Greens')
        plt.show()
    return 



# ===========================
#         USER INPUT 
# ===========================

#Nepsilon = np.array([16,16])
#NFine = np.array([256,256])
Nepsilon = np.array([64,64])
NFine = np.array([256,256])
NCoarse=np.array([32, 32])
k=1

alpha = 1.0
beta  = 10.0
p     = 0.1

incl_bl = np.array([0.25, 0.25])
incl_tr = np.array([0.75, 0.75])

Lshape_bl = np.array([0.5, 0.5])
Lshape_tr = np.array([0.75, 0.75])

shift_bl = np.array([0.75, 0.75])
shift_tr = np.array([1., 1.])

plot_offline_coeff(Nepsilon, NFine, NCoarse, k, alpha, beta, "erasure")

model3 = {'name':'inclshift', 'def_bl': shift_bl, 'def_tr': shift_tr}
model2 = {'name':'incllshape', 'def_bl': Lshape_bl, 'def_tr': Lshape_tr}

aPertList = [
    build_coefficient.build_inclusions_change_2d(
        NFine, Nepsilon, alpha, beta, incl_bl, incl_tr, p, model2
    ),
    build_coefficient.build_inclusions_change_2d(
        NFine, Nepsilon, alpha, beta, incl_bl, incl_tr, p, model3
    )
]

OUT_DIR = "data/Incl/"
os.makedirs(OUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

cmap = plt.get_cmap("viridis")
norm = matplotlib.colors.Normalize(vmin=alpha, vmax=beta)

for ax, aPert in zip(axes, aPertList):
    grid = aPert.reshape(NFine, order="C")
    im = ax.imshow(grid, origin="lower", extent=(0,1,0,1),
                   cmap=cmap, norm=norm)
    #ax.set_xticks([])
    #ax.set_yticks([])

fig.subplots_adjust(wspace=0.2)

cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.06)
#cbar.set_label("Coefficient value")

out_path = os.path.join(OUT_DIR, f"coefficients_side_by_side_p{p}.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")

aPert = build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p)
plots.aPertPlot(aPert, NFine, save_path=OUT_DIR)
plt.show()

#aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 0.5))
#aPertList.append(build_coefficient.build_inclusions_defect_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, 5.))
#aPertList.append(build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,incl_bl,incl_tr,p, model1))

