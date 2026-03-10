import numpy as np
import util

def build_inclusions_defect_2d(NFine, Nepsilon, bg, val, incl_bl, incl_tr, p_defect, def_val=None):        
    '''builds a fine coefficient which is periodic with periodicity length 1/epsilon. On the unit cell, the coefficient takes the value val inside a rectangle described  by following parameters.

    Parameters
    ----------
        incl_bl (bottom left) etc---> nodeal values?
        bg - background coefficient (The whites)
        val - inclusion value (The blacks) ---> small black blocks on relatively bigger white blocks ---> position can be chosen freely by defining incl_bl & incl_tr
        incl_bl -inclusion bottom left vertex value of the black block
        incl_tr - inclusion top right vertex value of the black block, otherwise the value is bg
        p_defect - probability of defect,  a defect means now, we erase the inclusion (the black box on the white block), with a probability of p_defect the inclusion 'vanishes', i.e. the value is set to def_val (default: bg)
        def_val -  This is to indicate if the black box is now grey, meaning, we don't entirely erase the black block to get a defect but its colour intensity reduces.
    '''

    assert(np.all(incl_bl) >= 0.)
    assert(np.all(incl_tr) <= 1.)
    assert(p_defect < 1.)

    if def_val is None:
        def_val = bg

    #probability of defect is p_defect
    c = np.random.binomial(1, p_defect, np.prod(Nepsilon))   

    aBaseSquare = bg*np.ones(NFine) # bg*([1, 1])
    flatidx = 0  
    for ii in range(Nepsilon[0]):          # Iterate through ε-blocks on the x-axis of the rectangle
        for jj in range(Nepsilon[1]):      # Iterate through ε-blocks on the y-axis of the rectangle
            startindexcols = int((ii + incl_bl[0]) * (NFine/Nepsilon)[0]) # ()* Number of fine-blocks per ε-block
            stopindexcols = int((ii + incl_tr[0]) * (NFine/Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine/Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine/Nepsilon)[1])
            if c[flatidx] == 0:
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
            else:
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = def_val
            flatidx += 1

    return aBaseSquare.flatten()  # 1-D array of aBaseSquare values



def build_inclusionbasis_2d(NPatch, NEpsilonElement, NFineElement, bg, val, incl_bl, incl_tr, defval=None):
    Nepsilon = NPatch * NEpsilonElement
    NFine = NPatch * NFineElement
    if defval is None:
        defval = bg

    assert (np.all(incl_bl) >= 0.)
    assert (np.all(incl_tr) <= 1.)

    aBaseSquare = bg * np.ones(NFine)
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine / Nepsilon)[1])
            aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val

    #aBase = aBaseSquare.flatten()

    def inclusion_defectI(ii):
        aSquare = np.copy(aBaseSquare)
        tmp_indx = np.array([ii % Nepsilon[1], ii // Nepsilon[1]])
        startindexcols = int((tmp_indx[0] + incl_bl[0]) * (NFine / Nepsilon)[0])
        stopindexcols = int((tmp_indx[0] + incl_tr[0]) * (NFine / Nepsilon)[0])
        startindexrows = int((tmp_indx[1] + incl_bl[1]) * (NFine / Nepsilon)[1])
        stopindexrows = int((tmp_indx[1] + incl_tr[1]) * (NFine / Nepsilon)[1])
        aSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = defval
        return aSquare.flatten()

    coeffList = list(map(inclusion_defectI, range(np.prod(Nepsilon))))
    coeffList.append(aBaseSquare.flatten())

    return coeffList

def build_inclusions_change_2d(NFine, Nepsilon, bg, val, incl_bl, incl_tr, p_defect, model):
    # builds a fine coefficient which is periodic with periodicity length 1/epsilon.
    # On the unit cell, the coefficient takes the value val inside a rectangle described by  incl_bl (bottom left) and
    # incl_tr (top right), otherwise the value is bg
    # with a probability of p_defect the inclusion 'changes', where three models are implemented:
    #    -filling the whole scaled unit cell (fill)
    #    -shifting the inclusion to def_bl, def_br
    #    - L-shape, i.e. erasing only the area def_bl to def_br

    assert(np.all(incl_bl) >= 0.)
    assert(np.all(incl_tr) <= 1.)
    assert(p_defect < 1.)

    assert(model['name'] in ['inclfill', 'inclshift', 'incllshape'])

    #probability of defect is p_defect
    c = np.random.binomial(1, p_defect, np.prod(Nepsilon))

    aBaseSquare = bg*np.ones(NFine)
    flatidx = 0
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine/Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine/Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine/Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine/Nepsilon)[1])
            if c[flatidx] == 0: # no defect
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
            else:
                if model['name'] == 'inclfill':
                    startdefindexcols = int((ii) * (NFine / Nepsilon)[0])
                    stopdefindexcols = int((ii + 1) * (NFine / Nepsilon)[0])
                    startdefindexrows = int((jj) * (NFine / Nepsilon)[1])
                    stopdefindexrows = int((jj + 1) * (NFine / Nepsilon)[1])
                    aBaseSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
                if model['name'] == 'inclshift':
                    def_bl = model['def_bl']
                    def_tr = model['def_tr']
                    startdefindexcols = int((ii + def_bl[0]) * (NFine / Nepsilon)[0])
                    stopdefindexcols = int((ii + def_tr[0]) * (NFine / Nepsilon)[0])
                    startdefindexrows = int((jj + def_bl[1]) * (NFine / Nepsilon)[1])
                    stopdefindexrows = int((jj + def_tr[1]) * (NFine / Nepsilon)[1])
                    aBaseSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
                if model['name'] == 'incllshape':
                    #first, put a normal inclusion
                    aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
                    # erase now the complement of the Lshape in the inclusion
                    def_bl = model['def_bl']
                    def_tr = model['def_tr']
                    startdefindexcols = int((ii + def_bl[0]) * (NFine / Nepsilon)[0])
                    stopdefindexcols = int((ii + def_tr[0]) * (NFine / Nepsilon)[0])
                    startdefindexrows = int((jj + def_bl[1]) * (NFine / Nepsilon)[1])
                    stopdefindexrows = int((jj + def_tr[1]) * (NFine / Nepsilon)[1])
                    aBaseSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = bg
            flatidx += 1

    return aBaseSquare.flatten()


def build_inclusionbasis_change_2d(NPatch, NEpsilonElement, NFineElement, bg, val, incl_bl, incl_tr, model):
    Nepsilon = NPatch * NEpsilonElement
    NFine = NPatch * NFineElement

    assert (np.all(incl_bl) >= 0.)
    assert (np.all(incl_tr) <= 1.)
    assert(model['name'] in ['inclfill', 'inclshift', 'incllshape'])

    aBaseSquare = bg * np.ones(NFine)
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine / Nepsilon)[1])
            aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val

    #aBase = aBaseSquare.flatten()

    def inclusion_defectI(ii):
        aSquare = np.copy(aBaseSquare)
        tmp_indx = np.array([ii % Nepsilon[1], ii // Nepsilon[1]])
        if model['name'] == 'inclfill':
            startdefindexcols = int((tmp_indx[0]) * (NFine / Nepsilon)[0])
            stopdefindexcols = int((tmp_indx[0] + 1) * (NFine / Nepsilon)[0])
            startdefindexrows = int((tmp_indx[1]) * (NFine / Nepsilon)[1])
            stopdefindexrows = int((tmp_indx[1] + 1) * (NFine / Nepsilon)[1])
            aSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
        if model['name'] == 'inclshift':
            def_bl = model['def_bl']
            def_tr = model['def_tr']
            #first erase the inclusion
            startindexcols = int((tmp_indx[0] + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((tmp_indx[0] + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((tmp_indx[1] + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((tmp_indx[1] + incl_tr[1]) * (NFine / Nepsilon)[1])
            aSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = bg
            #now put the inclusion at the new place
            startdefindexcols = int((tmp_indx[0] + def_bl[0]) * (NFine / Nepsilon)[0])
            stopdefindexcols = int((tmp_indx[0] + def_tr[0]) * (NFine / Nepsilon)[0])
            startdefindexrows = int((tmp_indx[1] + def_bl[1]) * (NFine / Nepsilon)[1])
            stopdefindexrows = int((tmp_indx[1] + def_tr[1]) * (NFine / Nepsilon)[1])
            aSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
        if model['name'] == 'incllshape':  # erase the complement of the Lshape in the inclusion
            def_bl = model['def_bl']
            def_tr = model['def_tr']
            startdefindexcols = int((tmp_indx[0] + def_bl[0]) * (NFine / Nepsilon)[0])
            stopdefindexcols = int((tmp_indx[0] + def_tr[0]) * (NFine / Nepsilon)[0])
            startdefindexrows = int((tmp_indx[1] + def_bl[1]) * (NFine / Nepsilon)[1])
            stopdefindexrows = int((tmp_indx[1] + def_tr[1]) * (NFine / Nepsilon)[1])
            aSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = bg
        return aSquare.flatten()

    coeffList = list(map(inclusion_defectI, range(np.prod(Nepsilon))))
    coeffList.append(aBaseSquare.flatten())

    return coeffList

