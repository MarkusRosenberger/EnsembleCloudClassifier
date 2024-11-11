#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mrosenberger
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#########################################################################################################################################################

def MLCM(ground_truth, predicted_classes, thresh = 0.5, return_plot = 'all', types = None, from_logits = False, **kwargs):
    '''

    Calculation of the Multi-Label ClassificatioN Matrix as defined by Heydarian et al. (2022)

    Parameters
    ----------
    ground_truth : 2d-array, shape == (n_instances, n_classes)
        Matrix holding the ground-truth observations one-hot encoded.
    predicted_classes : 2d-array, shape == (n_instances, n_classes)
        Matrix holding the model predictions, either probabilities or binary.
    thresh : float or array-like, optional
        Threshold probability for a prediction to be considered as true prediction.
        Can be a single value valid for each class or an array-like object with length "n_classes"
        if each class has a specific value. The default is 0.5.
    return_plot : string, optional
        Defines which plot is to be returned. Either "None" or "absolute" or "precision" or "recall" or "all". The default is 'all'.
    types : array-like, optional
        Array-like object containing the class-names to be used as tick parameters in plot. The default is None.
    from_logits : Boolean, optional
        Stating if the argument "predicted_classes" consists of logits. If "True" sigmoid activation is applied.
        If "False" nothing happens. The default is False.
    **kwargs : keyword(s)
        Additional keyword input for the plotting routine, e.g. cmap = 'Reds'.

    Returns
    -------
    2d-array
        The calculated values of the Multi-Label Confusion Matrix.
    Figure
        If stated by "return_plot" a plot of the MLCM of the corresponding measure is returned.
    '''
    
    # check shapes of ground truth and predictions
    assert np.shape(ground_truth) == np.shape(predicted_classes)
    
    # one probability threshold for all classes
    if type(thresh) is float:
        thresh = [thresh]*np.shape(ground_truth)[-1]

    # one probability threshold per class
    elif type(thresh) is list:
        thresh = np.array(thresh)
    
    # convert logits to probabilities if necessary
    # sigmoid activation for multi-label prediction
    if from_logits == True:
        predicted_classes = tf.keras.activations.sigmoid(predicted_classes)
    
    num_classes = np.shape(ground_truth)[1] # number of classes
    MLCM = np.zeros(shape = (num_classes+1, num_classes+1)) # empty MLCM matrix
    
    # filling the MLCM according to Heydarian et al. (2022)
    for y_true, y_pred in zip(ground_truth, predicted_classes):
        
        T = np.where(y_true == 1)[0]
        P = np.where(y_pred >= thresh)[0]
        
        T1 = []
        T2 = []
        P2 = []
        
        for p in P:
            if p in T:
                T1.append(p)
            else:
                P2.append(p)
        
        
        for t in T:
            if t not in T1:
                T2.append(t)
                
        # Step 1 (same for all categories)
        for t in T1:
            MLCM[t, t] += 1
        
        if len(P) == 0 and len(T) == 0:
            MLCM[num_classes, num_classes] += 1
        
        
        if P2 == []:
            # Step 2 of category 1
            for t in T2:
                MLCM[t, num_classes] += 1
                
        elif T2 == [] and P2 != []:
            # Step 2 of category 2
            for p in P2:
                if len(T) == 0:
                    MLCM[num_classes, p] += 1 
                else:
                    for t in T: 
                        MLCM[t, p] += 1 
                        
        elif T2 != [] and P2 != []:
            # Step 2 of category 3
            for p in P2:
                for t in T2: 
                    MLCM[t, p] += 1 
    
    # return only MLCM ...
    if return_plot == 'None' or return_plot == False:
        return MLCM
    # ... or also plot
    else:
        assert types is not None
        types = list(types)
        obs_per_class = np.sum(ground_truth,axis = 0)
        obs_per_class = list(obs_per_class) + [np.sum(MLCM[-1])]
        
        # which kind of plot is to be returned
        if return_plot == 'all':
            fig1 = plot_MLCM(MLCM, return_plot = 'absolute', num_classes = num_classes, types = types, obs_per_class = obs_per_class, **kwargs)
            fig2 = plot_MLCM(MLCM, return_plot = 'recall', num_classes = num_classes, types = types, **kwargs)
            fig3 = plot_MLCM(MLCM, return_plot = 'precision', num_classes = num_classes, types = types, **kwargs)
            return MLCM, fig1, fig2, fig3
        
        else:
            fig = plot_MLCM(MLCM, return_plot, num_classes, types, obs_per_class, **kwargs)
            return MLCM, fig

#########################################################################################################################################################

def plot_MLCM(MLCM, return_plot = 'absolute', num_classes = 30, types = [0,1,2,3,4,5,6,7,8,9]*3, obs_per_class = None, **kwargs):
    '''

    Plot the Multi-Label Classification Matrix as defined by Heydarian et al. (2022)

    Parameters
    ----------
    MLCM : 2d-array, shape == (num_classes+1, num_classes+1)
        Multi-Label Classification Matrix.
    return_plot : string, optional
        Defines which plot is to be returned. Either "absolute" or "precision" or "recall". The default is 'absolute'.
    num_classes : float
        How many classes the ground truth contains. The default is 30.
    types : array-like, optional
        Array-like object containing the class-names to be used as tick parameters in plot. The default is [0,1,2,3,4,5,6,7,8,9]*3 indicating 10 classes per height level.
    obs_per_class : array-like, optional
        Necessary only if return_plot = "absolute" to adapt cmap class-wise. The default is None.
    **kwargs : keyword(s)
        Additional keyword input for the plotting routine, e.g. cmap = 'Reds'.

    Returns
    -------
    Figure
        Plot of the MLCM, type is defined via return_plot argument.
    ''' 

    # initiate colormap, 'Reds' is used if no cmap is specified
    try:
        cmap = plt.get_cmap(kwargs['cmap'])
    except:
        cmap = plt.get_cmap('Reds')
    
    cmap.set_bad('white', alpha = 1)
    
    # define some parameters for the plot
    y,x = np.mgrid[-.5:num_classes+1.5,-.5:num_classes+1.5]
    xticks = np.arange(num_classes+1)
    yticks = np.arange(num_classes+1)[::-1]
    
    lw = 1
    fs = 8.4
    
    fig, ax = plt.subplots(1,1, figsize = (num_classes/4,num_classes/4))
    
    # MLCM of absolute values
    if return_plot == 'absolute':
        for i in range(num_classes+1):
            mask= np.ones(shape = np.shape(MLCM)).astype(bool)
            mask[i] = 0
            mask[MLCM == 0] = 1
            MLCM_masked = np.copy(MLCM)
            MLCM_masked[mask] = np.nan
            MLCM_plot = np.ma.masked_invalid(MLCM_masked)
            
            ax.pcolormesh(x, y[::-1], MLCM_plot, edgecolor = 'k', vmin = 0, vmax = obs_per_class[i]*1.2, lw = lw - .7, **kwargs)
    
        for r in range(num_classes+1):
            for c in range(num_classes+1):
                if int(MLCM[r,c]) == 0:
                    pass
                else:
                    plt.text(c, yticks[r], int(MLCM[r,c]), ha = 'center', va = 'center', fontsize = fs-2)
    
    # MLCM of class-wise recall values, i.e. row-wise normalized
    elif return_plot == 'recall':
        rowsum = np.sum(MLCM, axis = 1)
        rowsum[rowsum == 0] = 1
        
        recall_MLCM = np.array([row/rowsum[r] for r, row in enumerate(MLCM)])
    
        masked_recall_MLCM = np.copy(recall_MLCM)
        masked_recall_MLCM[recall_MLCM <= 0.005] = np.nan
    
        ax.pcolormesh(x, y[::-1], masked_recall_MLCM, edgecolor = 'k', vmin = 0, vmax = 1, lw = lw - .7, cmap = cmap)
        
        for r in range(num_classes+1):
            for c in range(num_classes+1):
                if round(recall_MLCM[r,c]*100) == 0:
                    pass
                else:
                    plt.text(c, yticks[r], '{:.0f}'.format(recall_MLCM[r,c]*100), ha = 'center', va = 'center', fontsize = fs-2)
    

    # MLCM of class-wise recall values, i.e. column-wise normalized
    elif return_plot == 'precision':
        colsum = np.sum(MLCM, axis = 0)
        colsum[colsum == 0] = 1
    
        precision_MLCM = np.array([row/colsum for row in MLCM])
        
        masked_precision_MLCM = np.copy(precision_MLCM)
        masked_precision_MLCM[precision_MLCM <= 0.005] = np.nan
        
        ax.pcolormesh(x, y[::-1], masked_precision_MLCM, edgecolor = 'k', vmin = 0, vmax = 1, lw = lw - .7, cmap = cmap)
        for r in range(num_classes+1):
            for c in range(num_classes+1):
                if round(precision_MLCM[r,c]*100) == 0:
                    pass
                else:
                    plt.text(c, yticks[r], '{:.0f}'.format(precision_MLCM[r,c]*100), ha = 'center', va = 'center', fontsize = fs-2)   
        
    # define ticklabels etc.
    ax.set_xticks(xticks)
    ax.set_xticklabels(types + ['NPL'], fontsize = fs-1)
    ax.set_xlabel('Predicted class', fontsize = fs)
    ax.set_yticks(yticks)
    ax.set_yticklabels(types + ['NTL'], fontsize = fs-1)
    ax.set_ylabel('True class', fontsize = fs)
    
    ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True,
                   left = True, labelleft = True, right = True, labelright = True)
    
    # indicating borders between height levels and labeling them
    if num_classes == 30:
        plt.hlines(y = [.5, 10.5, 20.5], xmin = -.5, xmax = 30.5, colors = 'blue', lw = lw)
        plt.vlines(x = [9.5, 19.5, 29.5], ymin = .5, ymax = 30.5, colors = 'blue', lw = lw)
        
        plt.text(x = 5, y = 32.2, s = r'C$_L$', ha = 'center', va = 'center', fontsize = fs)
        plt.text(x = 15, y = 32.2, s = r'C$_M$', ha = 'center', va = 'center', fontsize = fs)
        plt.text(x = 25, y = 32.2, s = r'C$_H$', ha = 'center', va = 'center', fontsize = fs)
        
        plt.text(x = 32.2, y = 5, s = r'C$_H$', ha = 'center', va = 'center', fontsize = fs)
        plt.text(x = 32.2, y = 15, s = r'C$_M$', ha = 'center', va = 'center', fontsize = fs)
        plt.text(x = 32.2, y = 25, s = r'C$_L$', ha = 'center', va = 'center', fontsize = fs)
    
    # black border lines for boxes on main-diagonal
    # x-coordinate starts on left side of figure
    # y-coordinate starts on bottom of figure
    
    # starting with first and last box
    # horizontal lines ...
    plt.hlines(x[0,-1]- 1e-3, xmin = x[0,0], xmax = x[0,0]+1, colors = 'k', lw = lw)
    plt.hlines(-.5, xmin = x[0,-1]-1, xmax = x[0,-1], colors = 'k', lw = lw)
    
    # ... and vertical lines
    plt.vlines(x[0,-1]- 1e-3, ymin = x[0,0], ymax = x[0,0]+1, colors = 'k', lw = lw)
    plt.vlines(-.47, ymin = x[0,-1]-1, ymax = x[0,-1], colors = 'k', lw = lw)
    
    # all other boxes in between
    for xi, yi in zip(x[0][:-2], y[1:-1,0][::-1]):
        plt.hlines(yi, xmin = xi, xmax = xi+2, colors = 'k', lw = lw)
        plt.vlines(yi, ymin = xi, ymax = xi+2, colors = 'k', lw = lw)
        
    return fig


#########################################################################################################################################################

def ReliabilityDiagram(y_pred, y_true, n_bins, title_string = None, return_plot = True, fs = 12):
    '''

    Parameters
    ----------
    y_pred : array
        Array containing the predicted probabilities of a given class at each instance.
    y_true : array
        Array containing one-hot encoded observation information.
    n_bins : int
        Number of bins the range [0,1] shall be divided into. 10 or 20 are the most common choices.
    title_string : str, optional
        String that should be the title of the plot, e.g. name of the class. The default is None.
    return_plot: bool, optional
        Defines if the reliability diagram should be returned. The default is True.
    fs: int, optional
        Fontsize in output plot. The default is 12.
        
    Returns
    -------
    Reliability, Resolution, Uncertainty: Directory
        The directory contains the values of these three statistics according to Murphy (1973).

    fig : Figure, optional
        Output figure. Upper subplot contains Reliability Diagram. 
        Lower one contains a bar chart of the relative frequency of occurence of each probability bin. 

    '''

    freq_clim = np.sum(y_true)/len(y_true) # climatological frequeny of observation of the class
    bin_borders = np.linspace(0,1,n_bins+1) # borders between probability bins
    bin_centers = [np.mean([bin_borders[i], bin_borders[i+1]]) for i in range(len(bin_borders)-1)] # centers of probability bins

    # empty arrays to fill
    bin_predictions = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_observations = np.zeros(n_bins)

    for pi, p in enumerate(y_pred):

        # find probability bin in which the prediction belongs        
        for index, value in enumerate(bin_borders[:-1]):
            # lower and upper bin border
            lower_border = value
            upper_border = bin_borders[index+1]
        
            if lower_border < p <= upper_border:
                bin_predictions[index] += 1 # count predictions in each bin
                bin_confidences[index] += p # sum of probabilities in each bin 
                bin_observations[index] += y_true[pi] # check if class observed or not in this instance
                break

    bin_mean_confidences = bin_confidences/bin_predictions # mean predicted probability in each bin
    bin_mean_accuracies = bin_observations/bin_predictions # frequency of observation for each bin
    bin_rel_predictions = bin_predictions/np.sum(bin_predictions) # relative number of predictions in each bin

    REL = np.nansum(bin_rel_predictions* (bin_mean_confidences - bin_mean_accuracies)**2) # reliability, smaller is better
    RES = np.nansum(bin_rel_predictions* (bin_mean_accuracies - freq_clim)**2) # resolution, larger is better
    UNC = freq_clim * (1-freq_clim) # uncertainty, just property of class under investigation, not of forecast model

    # no plot is returned
    if return_plot == False:
        return {'Reliability':REL, 'Resolution':RES, 'Uncertainty':UNC}

    else:
        # Plotting
        axesticks = np.linspace(0., 1, 6, dtype = 'float16')
        
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize = (5.3,8),height_ratios=[3, 1], sharex = True)
        fig.subplots_adjust(hspace = 0.05)
    
        # reliability diagram
        ax0.set(aspect = 1)
        ax0.plot(bin_mean_confidences, bin_mean_accuracies, c= 'k', marker = '.', markersize = 10, lw = 1.25)
        ax0.plot(bin_borders, bin_borders, ls = '--', c = 'k', lw = 1)
        ax0.plot(bin_borders, (bin_borders+freq_clim)/2, ls = '--', c = 'k', lw = .5)
        ax0.axhline(freq_clim, c = 'k', ls= '--', lw = .5)
        ax0.axvline(freq_clim, c = 'k', ls= '--', lw = .5)


        ax0.fill_between([0,freq_clim, freq_clim, 1],[0,0,1,1],[freq_clim/2.,freq_clim,freq_clim,1-(1-freq_clim)/2.], facecolor='lightgray', alpha=0.5) # shaded area for skill
        
        ax0.text(x = .97, y = .07, s = r'BSS = %2.3f'%((RES-REL)/UNC), fontsize = fs-1, ha = 'right') # add value of BSS
        
        ax0.set_ylabel(r'Observed relative frequency, $\overline{o_i}$', fontsize = fs)
        ax0.set_title(title_string, fontsize = fs)
    
        ax0.set_xlim(0,1)
        ax0.set_ylim(0,1)
    
        ax0.set_yticks(axesticks, axesticks, fontsize = fs)
    
        # relative number of predictions in each bin as bar chart
        ax1.bar(bin_centers, bin_rel_predictions, width = 1/n_bins, edgecolor = 'k')      
        ax1.set_ylabel('Density', fontsize =  fs)
        ax1.set_yticks(np.array(ax1.get_yticks()[:-1], dtype = 'float16'), np.array(ax1.get_yticks()[:-1], dtype = 'float16'), fontsize = fs)

        ax1.set_xlabel(r'Forecast probability, $y_i$', fontsize = fs)
        ax1.set_xlim(0,1)
        ax1.set_xticks(axesticks, axesticks, fontsize = fs)
        
        return fig, {'Reliability':REL, 'Resolution':RES, 'Uncertainty':UNC}

#########################################################################################################################################################

def Bootstrap_CM(TP, FP, FN, TN, n_draws = 10000):
    '''
    Draws a given number of randomly arranged permutations of a confidence matrix (CM) 
    according to Fowlkes et al. (1983) drawn from a hypergeometric distribution.

    Parameters
    ----------
    TP : int
        Number of True Positives in confidence matrix.
    FP : int
        Number of False Positives in confidence matrix.
    FN : int
        Number of False Negatives in confidence matrix.
    TN : int
        Number of True Negatives in confidence matrix.
    n_draws : int, optional
        Number of random draws, that should be made. The default is 10 000.
        
    Returns
    -------
    TP_bootstrap, FP_bootstrap, FN_bootstrap, TN_bootstrap: Directory
        The directory contains the arrays holding "n_draws" values of each of these four statistics.

    '''

    # calculate row- and column-wise sums of CM
    pos_obs = TP+FN # number of observations
    pos_pred = TP+FP # number of predictions
    neg_obs = FP+TN # number of non-observations
    neg_pred = FN+TN # number of non-predictions

    random_TP = []
    random_FP = []
    random_FN = []
    random_TN = []

    '''
    According to Fowlkes et al. (1983), one has to use a hypergeometric distribution. 
    np.random.hypergeometric needs 3 input variables:
        -) ngood: number of good selections
        -) nbad: number of bad selections
        -) nsample: number of items sampled

    for a given statistic:
        -) ngood is the sum of the row in which the statistic is located
        -) nbad = N - ngood, where N = TP + FP + FN + TN
        -) nsample is the sum of the column in which the statistic is located

    IMPORTANT: rows and columns are interchangeable in this context!
    '''

    for n in range(n_draws):
        # decide randomly which statistic is drawn first, otherwise biases could appear
        # and not all theoretically possible values are possibly drawn
        first_n = np.random.randint(0,4) # 0 -> TP, 1 -> FP, 2 -> FN, 3 -> TN
        
        # TP:
        if first_n == 0:
            # random draw with above described parameters
            draw_TP = np.random.hypergeometric(ngood = pos_obs, nbad = neg_obs, nsample = pos_pred)

            # calculate the other statistics given unchanged row- and column-wise sums
            draw_FN = pos_obs - draw_TP
            draw_FP = pos_pred - draw_TP
            draw_TN = neg_obs - draw_FP
        
        # FP:
        elif first_n == 1:
            draw_FP = np.random.hypergeometric(ngood = neg_obs, nbad = pos_obs, nsample = pos_pred)

            draw_TP = pos_pred - draw_FP
            draw_TN = neg_obs - draw_FP
            draw_FN = pos_obs - draw_TP
        
        # FN:
        elif first_n == 2:
            draw_FN = np.random.hypergeometric(ngood = pos_obs, nbad = neg_obs, nsample = neg_pred)

            draw_TP = pos_obs - draw_FN
            draw_TN = neg_pred - draw_FN
            draw_FP = neg_obs - draw_TN
        
        # TN:
        elif first_n == 3:
            draw_TN = np.random.hypergeometric(ngood = neg_obs, nbad = pos_obs, nsample = neg_pred)

            draw_FP = neg_obs - draw_TN
            draw_FN = neg_pred - draw_TN
            draw_TP = pos_obs - draw_FN

        random_TP.append(draw_TP)
        random_FP.append(draw_FP)
        random_FN.append(draw_FN)
        random_TN.append(draw_TN)

    random_TP = np.array(random_TP)
    random_FP = np.array(random_FP)
    random_FN = np.array(random_FN)
    random_TN = np.array(random_TN)

    # some final checks
    assert np.min((random_TP, random_FP, random_FN, random_TN)) >=0    
    assert np.all(random_TP + random_FN == pos_obs)
    assert np.all(random_TP + random_FP == pos_pred)
    assert np.all(random_FP + random_TN == neg_obs)
    assert np.all(random_FN + random_TN == neg_pred)

    return {'random_TP': random_TP, 'random_FP' : random_FP,
            'random_FN' : random_FN, 'random_TN' : random_TN}


#########################################################################################################################################################

def calculate_measures(MLCM, index, measures, do_bootstrap = True, **kwargs):
    '''
    Parameters
    ----------
    MLCM : (n+1) x (n+1) matrix
        (n+1) x (n+1) matrix with n being the number of classes. 
        TP, FP, FN, TN are calculated from the MLCM
    index : int
        0 <= index <= n-1 to decide for which class measures are calculated.
    measures : list or array
        Defines which measures are calculated. Possible are "MCC", "Precision", "Recall".
        If measures = None all of the above are calculated. 
    do_bootstrap : boolean, optional
        States if bootstrap sampling of TP, FP, FN, TN shall be made. The default is True.
    **kwargs : int
        Only possible keyword is "n_draws" to state how large the bootstrap sample has to be. Default is 10 000.

    Returns
    -------
    dict
        Dictionary holding all the measures in alphabetical order.
        If "do_bootstrap" == True after each measure all respective bootstrapped values are given 
        with key "bt_" + *name of measure*. 

    '''
    # read TP, FP, FN, TN from MLCM
    TP = MLCM[index, index]
    FN = np.sum(MLCM[index])-TP
    FP = np.sum(MLCM[:,index])-TP

    main_diagonal = np.eye(len(MLCM))*MLCM
    TN = np.sum(main_diagonal)-TP

    # calculate measures
    if len(measures) == 0:
        print('No measures are given, continue with default measures!')
        measures = ['MCC', 'Precision', 'Recall']

    # create random rearrangements of the collapsed MLCM
    if do_bootstrap == True:
        bt = Bootstrap_CM(TP, FP, FN, TN, **kwargs)
        
        bt_TP = bt['random_TP']
        bt_FP = bt['random_FP']
        bt_FN = bt['random_FN']
        bt_TN = bt['random_TN']

        measures_keys = np.array([[m, 'bt_'+m] for m in measures]).flatten()

    else:
        measures_keys = measures

    # create output directory with dummy values
    output_dict = {key: -9999 for key in measures_keys}

    # calculate MCC
    if 'MCC' in measures:
        num = TP*TN - FP*FN
        denom = np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))

        # Chicco et al. (2020) showed that 
        #   -) MCC -> 0 if a row- or column-wise sum of the confidence matrix equals 0
        #   -) MCC -> 1 if only one entry of the matrix is > 0, i.e. the whole sample is located in a single box
        # only the first case is considered here, since the second one does not occur anyway

        if denom == 0:
            output_dict['MCC'] = ( 0 )
        else:
            output_dict['MCC'] = ( num/denom )

        if do_bootstrap == True:
            bt_num = bt_TP*bt_TN - bt_FP*bt_FN
            bt_denom = np.sqrt((bt_TP+bt_FP) * (bt_TP+bt_FN) * (bt_TN+bt_FP) * (bt_TN+bt_FN))

            output_dict['bt_MCC'] = ( bt_num/bt_denom )

    # calculate Precision
    if 'Precision' in measures:
        output_dict['Precision'] = ( TP / (TP + FP) )
        
        if do_bootstrap == True:
            output_dict['bt_Precision'] = ( bt_TP / (bt_TP + bt_FP) )

    # calculate Recall
    if 'Recall' in measures:
        output_dict['Recall'] = ( TP / (TP + FN) )
        
        if do_bootstrap == True:
            output_dict['bt_Recall'] = ( bt_TP / (bt_TP + bt_FN) )        

    return output_dict

#########################################################################################################################################################