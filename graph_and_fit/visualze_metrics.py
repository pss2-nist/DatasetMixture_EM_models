import itertools
import os
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statannotations.Annotator import Annotator


# from scipy.stats import mannwhitneyu


def get_best_model_df(df, metric=None, groupby=None):
    optimal_product = None

    if groupby is None:
        groupby = ['channel', 'lr', 'pretrained']
    if metric is None:
        metric = "crossentropy"
    if metric in ["CE", "crossentropy"]:
        df['optimal_metric'] = np.sqrt(df['train loss'] ** 2 + df['validation loss'] ** 2)
        optimal_product = df.groupby(groupby)['optimal_metric'].idxmin()  # .reset_index()
    elif metric == "dice_macro":
        df['optimal_metric'] = df["dice_macro"]
        optimal_product = df.groupby(groupby)['optimal_metric'].idxmax()  # .reset_index()
    # print(f"minmetric { df}")
    print(f"minmetric {df['optimal_metric']}")
    # best_id_ch = df['optimal_metric'].idxmin()
    best_df_ch = df.loc[optimal_product]
    # minloss = df.min()
    return best_df_ch


def do_stat_annotations(df, ax, pair_param, x=None, y=None, order=None, pair_types="unique"):
    """
    :param df
    :param ax
    :param pair_param
    :param x
    :param y
    :param order
    :param pair_types: Select one from "unique, "adjacent"

    """
    try:
        pairs = None
        if pair_param == 'channel':
            pair_param_uniques = list(df.channel.unique())
        if pair_param == "pretrained":
            pair_param_uniques = list(df.pretrained.unique())
        if pair_param == "lr":
            pair_param_uniques = list(df.lr.unique())
        # pairs = list(set([(a, b) for a in pair_param_uniques for b in pair_param_uniques if a != b]))
        all_pairs = list(combinations(pair_param_uniques, 2))
        # print(all_pairs)
        # pairs = list(set(tuple(sorted(all_pair)) for all_pair in all_pairs))
        # print("ANNOTATING\n", df[df.isna().any(axis=1)], "\n", df.dtypes)
        annotator = Annotator(ax, all_pairs, data=df, x=x, y=y, order=order)
        # TODO: Add custom pairs
        # # for pair in pairs:
        # #     g1, g2 = p
        # print(pairs)
        # print(pairs[0])
        # print(pairs[0][1], type(pairs[0][1]))
        # print(ax.collections[0].get_array())
        # # print(df[df[pair_param] == pairs[0][1]])
        # exit()
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', line_height=0.005, text_offset=0,
                            fontsize=12)
        # annotator.configure(test='t-test_ind', text_format='star', loc='outside', line_height=0.01, text_offset=1)
        # annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        # annotator.apply_and_annotate()
        annotator.apply_test()
        # print(annotator.annotations[0])
        # exit()
        annotator.annotate()
        # annotator.apply_test(nan_policy='propagate')
        # annotator.annotate()
    except Exception as e:
        print("ERROR", e)
    return ax


def plot_metrics_comparison(df, savepath, comparemetrics=None, tr_thresh=100, tst_thresh=100, illustrate_thresh=True,
                            draw_lines=True):
    if comparemetrics is None:
        comparemetrics = ['train loss', 'validation loss']
    fig = plt.figure(1)
    metric1, metric2 = comparemetrics
    # tr_thresh = 100
    # tst_thresh = 100
    df = df[df['train loss'] <= tr_thresh]
    df = df[df['validation loss'] <= tst_thresh]
    if draw_lines:
        if metric1 in ['train loss', 'validation loss']:
            if metric2 in ['train loss', 'validation loss']:
                if illustrate_thresh == True:
                    x = [1, 2, 3, 4, 5]
                    y = [1, 2, 3, 4, 5]
                    plt.vlines(x, 0, y, linestyle="solid")
                    plt.hlines(y, 0, x, linestyle="solid")
    sns.scatterplot(df, x=metric1, y=metric2, s=5, alpha=0.95, hue=df['channel'],
                    palette='dark')  # label=f'{metric1}_vs_{metric2}', 'Spectral'
    plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"{metric1} vs {metric2}"
    plt.title(title_test, fontsize=20)
    plt.xlabel(metric1, fontsize=16)
    plt.ylabel(metric2, fontsize=16)
    outGraph_test = savepath + os.path.sep + f'{metric1}_vs_{metric2}_thresh{tr_thresh}-{tst_thresh}_comparison.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()


def get_stability(df, use_best=False, metric="CE", annotate_best_points=False, annotate_stability_stats=False):
    """
    Stability of a model is calculated  based on change of train and validation loss over the entire model.

    :param df: dataframe containing all data
    :param use_best: present stability for the best models

    """
    channels = list(df.channel.unique())
    lrs = list(df.lr.unique())
    pts = list(df.pretrained.unique())
    residual_arr_df = pd.DataFrame({'channel': pd.Series(dtype='str'),
                                    'lr': pd.Series(dtype='float64'),
                                    'pretrained': pd.Series(dtype='bool'),
                                    'm': pd.Series(dtype='float64'),
                                    'c': pd.Series(dtype='float64'),
                                    'residual': pd.Series(dtype='float64')})
    # print("INDEX ", residual_arr_df.index, residual_arr_df.columns)
    for channel in channels:
        df_ch = df[df['channel'] == channel]
        for lr in lrs:
            df_ch_lr = df_ch[df_ch['lr'] == lr]
            for pt in pts:
                df_ch_lr_pt = df_ch_lr[df_ch_lr['pretrained'] == pt]
                # find slope of training and test curves
                mintestloss = df_ch_lr_pt['validation loss'].min()
                mintrainloss = df_ch_lr_pt['train loss'].min()
                # df_testvals= df_ch_lr_pt[['epoch','validation loss']].to_numpy()
                df_trainvals, df_testvals = df_ch_lr_pt['train loss'].to_numpy(), df_ch_lr_pt[
                    'validation loss'].to_numpy()
                df_testvals = df_testvals - mintestloss
                df_trainvals = df_trainvals - mintrainloss
                A = np.vstack([df_trainvals, np.ones(len(df_trainvals))]).T
                [m, c], [tt_residual] = np.linalg.lstsq(A, df_testvals, rcond=None)[:2]
                residual_arr_df.loc[len(residual_arr_df), residual_arr_df.columns] = channel, lr, pt, m, c, tt_residual
                # r2_tt = 1 - tt_residual / (df_testvals.size * df_testvals.var())
                # print(m, c, tt_residual)
    # print(f"residual array: {residual_arr_df}")
    ######################################################################################
    qvals = [channels, lrs, pts]
    quants = ['channel', 'lr', 'pretrained']
    quantlabel = ['set of image modes', 'learning rates', 'pretrained']
    i = 0

    props = {
        'boxprops': {'facecolor': None, 'edgecolor': 'black'},
        # 'boxprops': {'edgecolor': 'Spectral'},
        'medianprops': {'color': 'green'},
        'whiskerprops': {'color': 'green'},
        'capprops': {'color': 'green'}
    }
    ####################################################################################################################
    ## Plotting residuals for all 48 models. Comparison between a secondary parameter included.
    ####################################################################################################################
    for quant in quants:
        for hue in quants:
            if hue != quant:
                fig = plt.figure(i)
                # print(f"residual array: {residual_arr_df}")
                plt.yscale('log')
                sns.boxplot(data=residual_arr_df, x=quant, y='residual', palette="Spectral", hue=hue,
                            showfliers=False, **props)
                plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
                sns.stripplot(data=residual_arr_df, x=quant, y='residual', color="black", dodge=5, hue=hue, size=3,
                              linewidth=0, legend=False)  # legend = False for newer versions
                # , alpha = 0.5
                plt.xticks(range(len(qvals[quants.index(quant)])), labels=qvals[quants.index(quant)])
                title_test = f"Residual wrt {quant}"
                # plt.title(title_test, fontsize=20)
                plt.xlabel(quantlabel[quants.index(quant)], fontsize=16)
                plt.ylabel('Residual', fontsize=16)
                outGraph_test = savepath + os.path.sep + f'Model_stability_{quant}_compare{hue}.png'
                plt.savefig(outGraph_test, bbox_inches='tight')
                plt.clf()
                plt.close()
                i += 1
    ####################################################################################################################
    ## Plotting residuals for all 48 models. Highlighted best model within each category.
    ####################################################################################################################
    props = {
        # 'boxprops': {'facecolor': 'none', 'edgecolor': 'red'},
        'boxprops': {'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }

    for quant in quants:
        fig = plt.figure(i)
        plt.yscale('log')
        # print("HIHI")
        if use_best:
            best_df_quant = get_best_model_df(df, groupby=[quant], metric=metric)
            columns_to_match = quants
            residual_arr_best = pd.merge(residual_arr_df, best_df_quant, on=columns_to_match, suffixes=('', '_best'))
            sns.stripplot(residual_arr_best, x=quant, y="residual", color="red", dodge=5, size=5, linewidth=0,
                          legend=False)
            # residual_arr_df = pd.merge(residual_arr_df, residual_arr_best, indicator=True).query(
            #     '_merge=="left_only"').drop('_merge', axis=1)
            for id, row in residual_arr_best.iterrows():
                # Show optimal metric instead of residual
                if annotate_best_points:
                    plt.gca().annotate(
                        f"{row['optimal_metric']:.4e}\n{row['channel']},\n{row['lr']}\n{row['pretrained']}",
                        (str(row[quant]), (row['residual'])), xytext=(20, -5),
                        textcoords='offset points',
                        family='sans-serif', fontsize=8, color='darkslategrey')
        sns.boxplot(residual_arr_df, x=quant, y="residual", palette="Spectral", showfliers=False, **props)
        # plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
        sns.stripplot(residual_arr_df, x=quant, y="residual", color="black", dodge=5, size=3,
                      linewidth=0, legend=False)
        residual_arr_df_na = residual_arr_df.dropna()

        if annotate_stability_stats:
            do_stat_annotations(df=residual_arr_df_na, ax=plt.gca(), pair_param=quant, x=quant, y="residual",
                                order=None)
        plt.xticks(range(len(qvals[quants.index(quant)])), labels=qvals[quants.index(quant)])
        # title_test = f"Residual wrt {quant}"
        # plt.title(title_test, fontsize=20)
        plt.xlabel(quantlabel[quants.index(quant)], fontsize=16)
        plt.ylabel('Residual', fontsize=16)
        outGraph_test = savepath + os.path.sep + f'Model_stability_{quant}_{metric}.png'
        plt.savefig(outGraph_test, bbox_inches='tight')
        plt.clf()
        plt.close()
        i += 1


def visualize_best_models(df, metric="CE", annotate_stats=False, annotate_best_points=False):
    """
        chooses best model based on RMS CrossEntropy metric
        selected for each training type
    """
    model_comparison_metrics = ["CE", "crossentropy", "dice_macro"]
    assert metric in model_comparison_metrics, f"metrics: {metric}"

    channels = list(df.channel.unique())
    lrs = list(df.lr.unique())
    pts = list(df.pretrained.unique())
    qvaltype = [channels, lrs, pts]
    quants = ['channel', 'lr', 'pretrained']
    quantlabel = ['set of image modes', 'learning rates', 'pretrained']

    i = 0
    best_df_all = get_best_model_df(df, groupby=quants, metric=metric)
    print("BEST\n", best_df_all)
    for qvals, quant in zip(qvaltype, quants):
        best_df_quant = get_best_model_df(df, groupby=[quant], metric=metric)
        fig = plt.figure(i)
        if metric in ["CE", "crossentropy"]:
            plt.yscale('log')
        # sns.boxplot(data=best_df_ch, x=quant, y='residual', palette="Spectral", showfliers=False)  # , **props)
        # plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
        sns.stripplot(data=best_df_all, x=quant, y='optimal_metric', color="black", dodge=5, size=3,
                      linewidth=0, legend=False)
        sns.stripplot(data=best_df_quant, x=quant, y='optimal_metric', color="red", dodge=5, size=5,
                      linewidth=0, legend=False)
        anotstring = ""
        if annotate_stats:
            do_stat_annotations(df=best_df_all, ax=plt.gca(), pair_param=quant, x=quant, y="optimal_metric", order=None)
            anotstring = "_wstat"
        for id, row in best_df_quant.iterrows():
            if annotate_best_points:
                plt.gca().annotate(f"{row['optimal_metric']:.4f}\n{row['channel']},\n{row['lr']}\n{row['pretrained']}",
                                   (str(row[quant]), (row['optimal_metric'])), xytext=(20, -5),
                                   textcoords='offset points',
                                   family='sans-serif', fontsize=8, color='darkslategrey')
        # title_test = f"Best models {quant}"
        # plt.title(title_test, fontsize=20)
        plt.xlabel(quantlabel[quants.index(quant)], fontsize=16)
        plt.ylabel(f'{metric}', fontsize=16)
        outGraph_test = savepath + os.path.sep + f'Best_model_{quant}_{metric}{anotstring}.png'
        plt.savefig(outGraph_test, bbox_inches='tight')
        plt.clf()
        plt.close()
        i += 1


def model_speed(df, use_best=True, metric="CE"):
    """
    Finds the number of epochs for a model that lie in ever smaller squares.
    For this metric, all epochs are taken into account.
    When comparing only the best models, the final best models for each category are identified
    and epochs are used only for those models.

    """
    channels = list(df.channel.unique())
    markers = ['.', 'v', '^', 'o', 's', "D", "P"]
    # print(channels, type(channels))
    thresholds = [5, 4, 3, 2, 1]
    # print(lims, type(lims))
    channel_model_speed = np.zeros((len(channels), len(thresholds)))
    best_df_quant = get_best_model_df(df, metric=metric)
    if use_best:
        df = pd.merge(df, best_df_quant, on=['channel', 'lr', 'pretrained'], suffixes=('', '_best'))
        # print(df)
    for channel in channels:
        df_ch = None
        for lim in thresholds:
            df_ch = df[df['channel'] == channel]
            df_ch = df_ch[df_ch['train loss'] <= lim]
            df_ch = df_ch[df_ch['validation loss'] <= lim]

            channel_model_speed[channels.index(channel), thresholds.index(lim)] = len(df_ch.index)
            # print(channel, lim, len(df_ch.index))
    fig = plt.figure(1)
    print(channel_model_speed.shape)
    for rc, rowc in enumerate(channel_model_speed):
        plt.plot(rowc, marker=markers[rc])

    plt.gca().invert_xaxis()

    plt.legend(channels, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"Model speed: values below threshold (higher is better)"
    # plt.title(title_test, fontsize=20)
    plt.xticks([0, 1, 2, 3, 4], labels=thresholds)
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200], labels=[0, 200, 400, 600, 800, 1000, 1200])
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('# Epoch values < threshold', fontsize=16)
    outGraph_test = savepath + os.path.sep + f'Model_convergence_speed_ch_{metric}{"_best" if use_best else ""}.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()
    ###############################################################################################################
    lrs = list(df.lr.unique())
    # print(channels, type(channels))
    thresholds = [5, 4, 3, 2, 1]
    # print(lims, type(lims))
    lr_model_speed = np.zeros((len(lrs), len(thresholds)))
    for lr in lrs:
        df_lr = None
        for lim in thresholds:
            df_lr = df[df['lr'] == lr]
            df_lr = df_lr[df_lr['train loss'] <= lim]
            df_lr = df_lr[df_lr['validation loss'] <= lim]
            lr_model_speed[lrs.index(lr), thresholds.index(lim)] = len(df_lr.index)
    fig = plt.figure(2)
    for rl, rowl in enumerate(lr_model_speed):
        plt.plot(rowl, marker=markers[rl])
    plt.plot()
    plt.gca().invert_xaxis()
    plt.legend(lrs, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"Model speed: values below threshold (higher is better)"
    # plt.title(title_test, fontsize=20)
    plt.xticks([0, 1, 2, 3, 4], labels=thresholds)
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200], labels=[0, 200, 400, 600, 800, 1000, 1200])
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('# Epoch values < threshold', fontsize=16)
    outGraph_test = savepath + os.path.sep + f'Model_convergence_speed_LR_{metric}{"_best" if use_best else ""}.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()
    ###############################################################################################################
    pts = list(df.pretrained.unique())
    # print(channels, type(channels))
    thresholds = [5, 4, 3, 2, 1]
    # print(lims, type(lims))
    pt_model_speed = np.zeros((len(pts), len(thresholds)))
    for pt in pts:
        df_pt = None
        for lim in thresholds:
            df_pt = df[df['pretrained'] == pt]
            df_pt = df_pt[df_pt['train loss'] <= lim]
            df_pt = df_pt[df_pt['validation loss'] <= lim]
            pt_model_speed[pts.index(pt), thresholds.index(lim)] = len(df_pt.index)
    fig = plt.figure(3)
    # plt.plot(pt_model_speed.T)
    for rp, rowp in enumerate(pt_model_speed):
        plt.plot(rowp, marker=markers[rp])

    plt.gca().invert_xaxis()

    plt.legend(pts, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"Model speed: values below threshold (higher is better)"
    # plt.title(title_test, fontsize=20)
    plt.xticks([0, 1, 2, 3, 4], labels=thresholds)
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               labels=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Pretrained with values below threshold', fontsize=16)
    outGraph_test = savepath + os.path.sep + f'Model_convergence_speed_Pretrained_{metric}{"_best" if use_best else ""}.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()
    # return channel_model_speed


if __name__ == "__main__":
    # path = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_PBS"

    # training_data = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_PBS/training_dic.xlsx"
    # savepath = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_PBS/figs_paper"
    training_data = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_DDS/training_dic.xlsx"
    savepath = "C:/Users/pss2/PycharmProjects/DatadrivenEM/data/measured_SEM_images/Synthetic_DDS/figs_paper"
    try:
        dataF = pd.read_csv(training_data)
    except:
        dataF = pd.read_excel(training_data)
    dataF.columns = dataF.columns.str.lower()

    allmetrics = ['train loss', 'validation loss', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro',
                  'per-pixel accuracy', 'dice_macro', 'dice_micro', 'jaccard_micro', 'jaccard_macro',
                  'confidence_sd_macro',
                  'confidence_sd_micro', 'mse']
    ##################################################################
    calcmetrics = True
    os.makedirs(savepath, exist_ok=True)
    try:
        dataF = dataF.replace('H0', '{H0}')
        dataF = dataF.replace('Hdark', '{DF}')
        dataF = dataF.replace('H0H1', '{H0,H1}')
        dataF = dataF.replace('H0Hdark', '{H0,DF}')
        dataF = dataF.rename(columns={'train_loss': 'train loss', 'test_loss': 'validation loss'})
    except Exception as renameerr:
        print(renameerr)
        exit()

    print(dataF.columns)
    # exit()
    if calcmetrics:
        get_stability(dataF, use_best=True, metric="crossentropy", annotate_best_points=False,
                      annotate_stability_stats=True)
        for met in ['dice_macro', 'crossentropy']:
            for ans in [True, False]:
                visualize_best_models(dataF, metric=met, annotate_best_points=False, annotate_stats=ans)
            # for usebest in [True]:
            model_speed(dataF, use_best=True, metric=met)
    ##################################################################

    visualize = True
    if visualize:
        for i in allmetrics:
            for j in allmetrics:
                if i != j:
                    plot_metrics_comparison(dataF, savepath=savepath, comparemetrics=[i, j], tr_thresh=5,
                                            tst_thresh=5, draw_lines=True)
