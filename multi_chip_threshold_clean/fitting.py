import scipy
import sinter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from lmfit import Parameters, minimize, report_fit, report_ci, conf_interval, Minimizer, fit_report

# %% Definition of ansatz models


def logical_error_model(x, pth, alpha, d):
    """Error model for a regular surface code."""
    return (alpha*(x/pth)**((d+1)/2))


def logical_error_model_pow(x, pth, alpha, beta, d):
    """Error model for a regular surface code with a log."""
    return (alpha*(x/pth)**(beta*(d+1)))


def log_logical_error_model(x, intercept, d):
    """Log version of the above."""
    return (intercept+x*((d+1)/2))


def decoupled_error_model(ps, pb, psth, pbth, alpha1, alpha2, d):
    """Logical error model with a seam.

    when we don't consider errors leaving the seam and coming back.
    Naive ansatz.
    """
    return (logical_error_model(ps, psth, alpha1, d)+logical_error_model(pb, pbth, alpha2, d))


def coupled_error_threshold(pb, psth, pbth, alphac):
    """Modify threshold by excursion in the bulk.

    Ansatz originally developed for unrotated surface code.
    """
    return (psth/(1+alphac*pb*np.sqrt(psth)/(1-np.sqrt(pb/pbth)))**2)


def coupled_error_threshold_v2(pb, psth, pbth, alphac):
    """Modify threshold by excursion in the bulk.

    Ansatz suited for rotated surface code.
    """
    return (psth/(1+alphac/(1-np.sqrt(pb/pbth)))**2)


def coupled_error_model(ps, pb, alpha1, alpha2, alpha3, psth, pbth, alphac, d):
    """Complete ansatz for unrotated surface code."""
    coupling_term = sum(((ps/coupled_error_threshold(pb, psth, pbth, alphac))
                        ** (i/2)*(pb/pbth)**((d+1-i)/2) for i in range(1, d+1)))
    return (decoupled_error_model(ps, pb, psth, pbth, alpha1, alpha2, d)+alpha3*coupling_term)


def coupled_error_model_v2(ps, pb, alpha1, alpha2, alpha3, psth, pbth, alphac, d):
    """Complete Ansatz for rotated lattice."""
    coupling_term = sum(((ps/coupled_error_threshold_v2(pb, psth, pbth, alphac))
                        ** (i/2)*(pb/pbth)**((d+1-i)/2) for i in range(1, d+1)))
    return (decoupled_error_model(ps, pb, psth, pbth, alpha1, alpha2, d) +
            alpha3*coupling_term)


def coupled_error_model_v3(ps, pb, alpha1, alpha2, alpha30, alpha31, alpha32, psth, pbth,
                           alphac, d):
    """Complete ansatz with modification due to rotated lattice.

    We include the possibility of having a polynomial prefactor in front of the interaction term.
    """
    coupling_term = sum(((ps/coupled_error_threshold_v2(pb, psth, pbth, alphac))
                        ** (i/2)*(pb/pbth)**((d+1-i)/2) for i in range(1, d+1)))
    return (decoupled_error_model(ps, pb, psth, pbth, alpha1, alpha2, d) +
            (alpha30+alpha31*d+alpha32*d**2)*coupling_term)


def multiple_seam_ansatz(ps, pb, alpha1, alpha2, alpha3, psth, pbth, alphac, d, nb_seam, length):
    """Complete ansatz formula for the case with several interfaces."""
    coupling_term = sum(((ps/coupled_error_threshold_v2(pb, psth, pbth, alphac))
                        ** (i/2)*(pb/pbth)**((d+1-i)/2) for i in range(1, d+1)))
    return (decoupled_error_model(ps, pb, psth, pbth, alpha1*nb_seam, alpha2*length/d, d) +
            alpha3*nb_seam*coupling_term)


def critical_exponent(p, pth, A, B, C, v, d):
    """Fit for critical exponent method around threshold."""
    return (A+B*(p-pth)*d**(1/v)+C*(p-pth)**2*d**(2/v))


# %% Tool functions


def nbr_cycle(d):
    """Return the number of cycle in one simulation."""
    return (3*d)


def stat_error(p, n, rep=1):
    """Statistical error for a binomial sampling."""
    return (np.sqrt(p*(1-p)/n)/(rep*(1-p)**(1-1/rep)))


def to_error_per_round(proba, nb):
    """Transform the logical error rate after d rounds to the logical error rate per rounds."""
    return (1-(1-proba)**(1/nb))


def getPairs(inList):
    """Give all the possible pairs."""
    outList = []
    for i in range(len(inList)):
        item = inList[0]
        inList = inList[1:]
        for j in inList:
            outList.append((item, j))

    return outList


def filter_data(data_tofit, pmin=0., pmax=1., psmin=0., psmax=1., ratio_std_err=100.):
    """Filter the data for the 3D fit.

    data_tofit is a list of dict of type data_tofit returned by
    prepare_data. each element of the list correspond to a fixed distance.
    """
    p_log = data_tofit[2][:]
    std_err = data_tofit[3][:]
    ps = data_tofit[1][:]
    p = data_tofit[0][:]
    index_to_remove = []
    for i in range(len(p)):
        if (p[i] < pmin) or (p[i] > pmax) or (ps[i] < psmin) or (ps[i] > psmax) or (
                std_err[i] > ratio_std_err*p_log[i]):
            index_to_remove.append(i)
    p_filtered = np.delete(p, index_to_remove)
    ps_filtered = np.delete(ps, index_to_remove)
    std_err_filtered = np.delete(std_err, index_to_remove)
    p_log_filtered = np.delete(p_log, index_to_remove)
    return (p_log_filtered, p_filtered, ps_filtered, std_err_filtered)
# %% Prepare the data in a readable dictionary format


def prepare_data(samples, data_type, fits=[]):
    """Prepare the dictionary used to do the fits.

    depends on different entry parameters
    (ratio, p fixe, p_bell fixe...).
    data_type can be 'no_split' or 'ratio' or 'p fixed' or 'p_bell fixed' or '3d'.
    fits list only matter for no split to choose between regular ansatz and critical exponent.
    """
    data_tofit = {}
    data = sinter.group_by(samples, key=lambda stat: stat.json_metadata['d'])
    if data_type == 'no split':
        if 'critical exponent' not in fits:
            # Fix the threshold for latter fitting, for the moment threshold is estimated by hand
            pth = 3*10**(-3)
            # pth = 2e-1
            # Prepare the data for the fits
            for d, s in data.items():
                data_tofit[d] = np.array([sum(([sample.json_metadata['p']] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)), []),
                    sum(([to_error_per_round(sample.errors/sample.shots,
                                             sample.json_metadata['k'])] for sample in s if (
                        sample.errors > 0 and sample.json_metadata['p'] < pth)
                    ), []),
                    sum(([stat_error(sample.errors/sample.shots, sample.shots,
                                     rep=sample.json_metadata['k'])] for sample in s if (
                        sample.errors > 0 and sample.json_metadata['p'] < pth)
                    ), [])
                ])
        elif 'critical exponent' in fits:
            # Fix the threshold for latter fitting, for the moment threshold is estimated by hand
            pth = 7*10**(-3)
            # Prepare the data for the fits
            for d, s in data.items():
                data_tofit[d] = np.array([sum(([sample.json_metadata['p']] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)), []),
                    sum(([sample.errors/sample.shots,] for sample in s if (
                        sample.errors > 0 and sample.json_metadata['p'] < pth)
                    ), []),
                    sum(([stat_error(sample.errors/sample.shots,
                                     sample.shots)] for sample in s if (
                        sample.errors > 0 and sample.json_metadata['p'] < pth)
                    ), [])
                ])

    elif data_type == 'ratio':
        # Fix the threshold for latter fitting, for the moment threshold is estimated by hand
        pth = 5*10**(-3)
        ratio = 43
        psth = ratio*pth
        # Prepare the data for the fits
        for d, s in data.items():
            data_tofit[d] = np.array([sum(([sample.json_metadata['p']] for sample in s if (
                sample.errors > 0 and sample.json_metadata['p'] < pth)), []),
                sum(([sample.errors/sample.shots] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)),
                    []),
                sum(([stat_error(sample.errors/sample.shots, sample.shots)]
                     for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)),
                    []),
                sum(([sample.json_metadata['p_bell']] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p_bell'] < psth)),
                    [])
            ])
    elif data_type == 'p fixed':
        # Fix the threshold for latter fitting, for the moment threshold is estimated by hand
        pth = 10**(-1)
        # Prepare the data for the fits
        for d, s in data.items():
            data_tofit[d] = np.array([sum(([sample.json_metadata['p_bell']] for sample in s if (
                sample.errors > 0 and sample.json_metadata['p_bell'] < pth)), []),
                sum(([to_error_per_round(sample.errors/sample.shots,
                                         sample.json_metadata['k'])] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p_bell'] < pth)),
                    []),
                sum(([stat_error(sample.errors/sample.shots, sample.shots,
                                 rep=sample.json_metadata['k'])] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p_bell'] < pth)),
                    []),
                sum(([sample.json_metadata['p']] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p_bell'] < pth)),
                    [])
            ])

    elif data_type == 'p_bell fixed':
        # Fix the threshold for latter fitting, for the moment threshold is estimated by hand
        pth = 5*10**(-3)
        # Prepare the data for the fits
        for d, s in data.items():
            data_tofit[d] = np.array([sum(([sample.json_metadata['p']] for sample in s if (
                sample.errors > 0 and sample.json_metadata['p'] < pth)), []),
                sum(([to_error_per_round(sample.errors/sample.shots,
                                         sample.json_metadata['k'])] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)),
                    []),
                sum(([stat_error(sample.errors/sample.shots, sample.shots,
                                 rep=sample.json_metadata['k'])] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)),
                    []),
                sum(([sample.json_metadata['p_bell']] for sample in s if (
                    sample.errors > 0 and sample.json_metadata['p'] < pth)),
                    [])
            ])
    elif data_type == '3d':
        for d, s in data.items():
            data_tofit[d] = np.array([sum(([sample.json_metadata['p']] for sample in s if (
                sample.errors > 0)), []),
                sum(([sample.json_metadata['p_bell']]
                     for sample in s if (sample.errors > 0)), []),
                sum(([to_error_per_round(sample.errors/sample.shots,
                                         sample.json_metadata['k'])]
                     for sample in s if (sample.errors > 0)), []),
                sum(([stat_error(sample.errors/sample.shots, sample.shots,
                                 rep=sample.json_metadata['k'])]
                     for sample in s if (sample.errors > 0)), [])
            ])
    return (data_tofit, data)


# %% Comparison of data with models via fit and/or plot
def plot_data_vs_model(data_tofit, data_type):
    """Plot Expected behavior with far apart multiple seams."""
    num_points = 10000
    alpha3 = 0.053
    alphac = 0.21
    alpha1 = 0.099
    alpha2 = 0.045
    pbth = 7.2e-3
    psth = 0.298
    nb_seam = 2
    plt.figure(figsize=(10, 6))
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow
        "#17becf"   # Cyan
    ]
    markers = [
        "o",  # Circle
        "v",  # Triangle Down
        "^",  # Triangle Up
        "s",  # Square
        "p",  # Pentagon
        "*",  # Star
        "+",  # Plus
        "x",  # X
    ]
    marker_index = 0
    color_index = 0
    for d in data_tofit.keys():
        p_log = data_tofit[d][1][:]
        std_err = data_tofit[d][2][:]
        p_s_or_b = data_tofit[d][0][:]
        color = colors[color_index % len(colors)]
        color_index += 1  # Increment the index
        marker = markers[marker_index % len(markers)]
        marker_index += 1  # Increment the index
        x_model = np.linspace(
            np.min(p_s_or_b), np.max(p_s_or_b), num_points)
        plt.errorbar(p_s_or_b, p_log, yerr=std_err,
                     fmt=marker, capsize=5, label=f'd={d}', color=color, ecolor=color)
        if data_type == 'p fixed':
            p = data_tofit[d][3][0]
            # Compute y values
            y1 = multiple_seam_ansatz(x_model, p, alpha1, alpha2, alpha3, psth,
                                      pbth, alphac, d, nb_seam, 4*d+3)
            # Plot the function
            plt.plot(x_model, y1, label=None, color=color, linestyle='--')
            # Adding labels and title
            plt.xlabel(r'$p_{\text{Bell}}$', fontsize=21)
            plt.ylabel(r'$P_L^X$', fontsize=21, rotation=0, ha='right')
            plt.xscale('log')
            plt.yscale('log')
            # Customize grid to show logarithmic scale on the background
            plt.grid(True, which="both", linestyle="-", alpha=0.6, linewidth=0.9)
            plt.tick_params(axis='both', which='both', labelsize=14)
            # Add vertical lines at p_Bell = 1e-2 and p_Bell = 5e-2
            plt.axvline(x=1e-2, color='black', linestyle=':')
            plt.axvline(x=5e-2, color='black', linestyle=':')
            plt.legend(fontsize=14)
        if data_type == 'p_bell fixed':
            p_bell = data_tofit[d][3][0]
            # Compute y values
            y1 = multiple_seam_ansatz(p_bell, x_model, alpha1, alpha2, alpha3, psth,
                                      pbth, alphac, d, nb_seam, 4*d+3)
            # Plot the function
            plt.plot(x_model, y1, label=None, color=color, linestyle='--')
            # Adding labels and title
            plt.xlabel(r'$p$')
            plt.ylabel(r'$p_L')
            plt.xscale('log')
            plt.yscale('log')
            # Customize grid to show logarithmic scale on the background
            plt.grid(True, which="major", linestyle="-", alpha=0.6, linewidth=0.9)
            plt.title("Multi seam logical error rate")
            plt.legend()
    plt.show()


def fit(nb_shots, data_tofit, data, fits, data_type, method):
    """Do the fit using a dict from prepare_data."""
    # General fits :

    def objective_werror(params):
        resid = []
        for d in list(data.keys()):
            if d > 2:
                p_log = data_tofit[d][1][:-1]
                p = data_tofit[d][0][:-1]
                pth = params['pth'].value
                alpha = params['alpha'].value
                std_err = (data_tofit[d][2][:-1])
                # Objective function is of the form (y-f(x))/std_y
                resid = np.concatenate((resid, list((p_log-logical_error_model(
                    p, pth, alpha, d)) / std_err)))
        return (resid)

    def objective_pow(params):
        resid = []
        for d in list(data.keys()):
            if d > 3:
                p_log = data_tofit[d][1][:-1]
                p = data_tofit[d][0][:-1]
                pth = params['pth'].value
                alpha = params['alpha'].value
                beta = params['beta'].value
                std_err = (data_tofit[d][2][:-1])
                # Objective function is of the form (y-f(x))/std_y
                resid = np.concatenate((resid, list(((p_log)-logical_error_model_pow(
                    p, pth, alpha, beta, d)) / std_err)))
        return (resid)

    def objective_critical(params):
        resid = []
        for d in list(data.keys()):
            # Objective function is of the form (y-f(x))/std_y to do the fit around
            # the threshold for precise threshold estimation
            if d >= 9:
                p_log = data_tofit[d][1][:-1]
                p = data_tofit[d][0][:-1]
                pth = params['pth'].value
                A = params['A'].value
                B = params['B'].value
                C = params['C'].value
                v = params['v'].value
                std_err = (data_tofit[d][2][:-1])
                resid = np.concatenate((resid, list((p_log-critical_exponent(p, pth,
                                       A, B, C, v, d))/std_err)))
        return (resid)

    if 'critical exponent' in fits:
        fit_params2 = Parameters()
        fit_params2.add('A', value=0, min=-100.0, max=100.0)
        fit_params2.add('B', value=0, min=-100.0, max=100.0)
        fit_params2.add('C', value=0, min=-1000.0, max=1000.0)
        fit_params2.add('pth', value=5e-3, min=0.0, max=1.0)
        fit_params2.add('v', value=1, min=0.0, max=2)
        out2 = minimize(objective_critical, fit_params2)
        print(fit_report(out2))
        # Plot the fit

        plt.yscale("log")
        plt.xscale("log")
        for d in data_tofit.keys():
            x_fit = np.linspace(
                np.min(data_tofit[d][0]), np.max(data_tofit[d][0]), 100)
            plt.errorbar(data_tofit[d][0], data_tofit[d][1],
                         yerr=data_tofit[d][2], fmt="o", label=f"d={d}", capsize=2)
            # plt.ylim((1/nb_shots)*1e-1,1e-1)  # Adjust the limits as needed
            plt.plot(x_fit, critical_exponent(x_fit, out2.params['pth'].value, out2.params['A'].value,
                     out2.params['B'].value, out2.params['C'].value, out2.params['v'].value, d), "r")

        plt.legend()
        plt.title(
            f"critical exponent fit gave pth={round(out2.params['pth'].value,5)} +-"
            f"{round(out2.params['pth'].stderr,7)}")
        plt.show()

    if 'general with error' in fits:
        fit_params2 = Parameters()
        fit_params2.add('alpha', value=1, min=0.0, max=100.0)
        fit_params2.add('pth', value=1e-2, min=0.0, max=1.0)
        # Create the Minimizer
        mini = Minimizer(objective_werror, fit_params2, nan_policy='propagate')
        if method == 'emcee':
            out2 = mini.minimize(objective_werror, fit_params2,
                                 method=method, is_weighted=True)
            print(fit_report(out2))
        if method == 'leastsq':
            # fit avec leastsq
            out2 = mini.minimize(method='leastsq')
            print(fit_report(out2))
            # Calcule de l'intervalle de confiance
            ci = conf_interval(mini, out2)
            report_ci(ci)

        # Plot the fit
        plt.close('all')
        plt.yscale("log")
        plt.xscale("log")
        for d in data_tofit.keys():
            # plt.plot(data_tofit[d][0], data_tofit[d][1],"o", label =f"d={d}")
            x_fit = np.linspace(
                np.min(data_tofit[d][0]), np.max(data_tofit[d][0]), 100)
            plt.errorbar(data_tofit[d][0], data_tofit[d][1],
                         yerr=data_tofit[d][2], fmt="o", label=f"d={d}", capsize=2)
            plt.ylim((1/nb_shots)*1e-1, 1e-1)  # Adjust the limits as needed
            plt.plot(x_fit, logical_error_model(
                x_fit, out2.params['pth'].value, out2.params['alpha'].value, d), "r")
            plt.fill_between(x_fit,
                             logical_error_model(
                                 x_fit, out2.params['pth'].value + out2.params['pth'].stderr,
                                 out2.params['alpha'].value - out2.params['alpha'].stderr, d),
                             logical_error_model(
                                 x_fit, out2.params['pth'].value - out2.params['pth'].stderr,
                                 out2.params['alpha'].value + out2.params['alpha'].stderr, d),
                             color='gray', alpha=0.2)
        plt.legend()
        plt.title(
            r"$\alpha=$"f"{round(out2.params['alpha'].value,3)} +-"
            f" {round(out2.params['alpha'].stderr,3)} and "r"$p_{\text{threshold}}$"
            f"={round(out2.params['pth'].value,4)} +- {round(out2.params['pth'].stderr,4)}")
        plt.show()

    if ('slopes' in fits) or ('slope+alpha' in fits) or ('crossing asymptots' in fits):

        # Store the data of the fits in a dictionary to later estimate the threshold
        Asymptots = {}
        # Define a list of colors for each distance d
        colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow
            "#17becf"   # Cyan
        ]
        markers = [
            "o",  # Circle
            "v",  # Triangle Down
            "^",  # Triangle Up
            "s",  # Square
            "p",  # Pentagon
            "*",  # Star
            "+",  # Plus
            "x",  # X
        ]
        color_idx = 0  # Index for color list
        marker_idx = 0  # Index for marker list
        for d in data_tofit.keys():
            p = data_tofit[d][0][:]
            p_log = data_tofit[d][1][:]
            std_err = data_tofit[d][2]
            fit = scipy.stats.linregress(np.log(p), np.log(p_log))
            Asymptots[d] = [fit.slope, fit.intercept]
            print(f"The slope for d={d} is {fit.slope}")
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel(r'$p_{\text{Bell}}$', fontsize=21)
            plt.ylabel(r'$P_L^X$', fontsize=21, rotation=0, ha='right')
            plt.ylim((1/nb_shots)*1e-2, 1e-1)  # Adjust the limits as needed
            # Assign a color and marker for the current d
            color = colors[color_idx]
            marker = markers[marker_idx]
            color_idx += 1
            marker_idx += 1
            # Plot points and fitted line with the same color and corresponding marker
            plt.errorbar(
                p, p_log, yerr=std_err, fmt=marker, color=color,
                label=f'd={d} and ' + r'$d_{eff}$' + f'={round(2*fit.slope-1, 2)}', capsize=2
            )
            x_fit = np.linspace(np.min(data_tofit[d][0]), np.max(data_tofit[d][0]), 100)
            plt.plot(
                x_fit, np.exp(log_logical_error_model(np.log(x_fit), fit.intercept, 2*fit.slope-1)), color=color
            )
        plt.tick_params(axis='both', which='both', labelsize=14)
        plt.grid(True, which="both", linestyle="-", alpha=0.6, linewidth=0.9)
        plt.legend(fontsize=14)
        plt.show()

    if 'two contributions' in fits:
        fit_params3 = Parameters()
        fit_params3.add('alpha2', value=1, min=0.0, max=100.0)
        fit_params3.add('psth', value=1e-2, min=0.0, max=1.0)
        alpha1 = 0.2168
        pbth = 4.7e-3
        if data_type == 'p_bell fixed':
            ps = data_tofit[list(data.keys())[0]][3][0]
        if data_type == 'p fixed':
            p = data_tofit[list(data.keys())[0]][3][0]
        if data_type == 'ratio':
            ratio = 43
        # define the residual function

        def objective_decoupled(params):
            resid = []
            for d in list(data.keys()):
                p_log = data_tofit[d][1][:-1]
                psth = params['psth'].value
                alpha2 = params['alpha2'].value
                std_err = data_tofit[d][2][:-1]
                # Objective function is of the form (y-f(x))/std_y
                if data_type == 'p_bell fixed':
                    p = data_tofit[d][0][:-1]
                    ps = data_tofit[d][3][:-1]
                if data_type == 'p fixed':
                    p = data_tofit[d][3][:-1]
                    ps = data_tofit[d][0][:-1]
                if data_type == 'ratio':
                    p = data_tofit[d][0][:-1]
                    ps = data_tofit[d][3][:-1]
                resid = np.concatenate((resid, list((p_log-decoupled_error_model(
                    ps, p, psth, pbth, alpha1, alpha2, d))/std_err)))
            return (resid)

        # Perform the fit
        out3 = minimize(objective_decoupled, fit_params3, method=method)
        print(fit_report(out3))

        # Plot the result with the fit
        plt.yscale("log")
        plt.xscale("log")
        for d in data_tofit.keys():
            p_log = data_tofit[d][1]
            std_err = data_tofit[d][2]
            psth = out3.params['psth'].value
            alpha2 = out3.params['alpha2'].value
            # p or ps depending of the type of data
            x_fit = np.linspace(
                np.min(data_tofit[d][0]), np.max(data_tofit[d][0]), 100)
            plt.errorbar(data_tofit[d][0], p_log,
                         yerr=std_err, fmt="o", label=f"d={d}", capsize=2)
            plt.ylim((1/nb_shots)*1e-1, 1e-1)  # Adjust the limits as needed
            if data_type == 'p_bell fixed':
                plt.plot(x_fit, decoupled_error_model(
                    ps, x_fit, psth, pbth, alpha1, alpha2, d), "r")
                plt.title(
                    f"Curves for the two contribution fit.\n alpha_bulk={round(out3.params['alpha2'].value,4)} +- {round(out3.params['alpha2'].stderr,5)} and p_seam_th={round(out3.params['psth'].value,5)} +- {round(out3.params['psth'].stderr,6)} \n Seam contribution is fixed to p_s={ps}, p_bulk_th={pbth}")
            if data_type == 'p fixed':
                plt.plot(x_fit, decoupled_error_model(
                    ps, x_fit, psth, pbth, alpha1, alpha2, d), "r")
                plt.title(
                    f"Curves for the two contribution fit.\n alpha_bulk={round(out3.params['alpha2'].value,4)} +- {round(out3.params['alpha2'].stderr,5)} and p_seam_th={round(out3.params['psth'].value,5)} +- {round(out3.params['psth'].stderr,6)} \n bulk contribution is fixed to p={p}, p_bulk_th={pbth}")
            if data_type == 'ratio':
                plt.plot(x_fit, decoupled_error_model(
                    ratio*x_fit, x_fit, psth, pbth, alpha1, alpha2, d), "r")
                plt.title(
                    f"Curves for the two contribution fit.\n alpha_bulk={round(out3.params['alpha2'].value,4)} +- {round(out3.params['alpha2'].stderr,5)} and p_seam_th={round(out3.params['psth'].value,5)} +- {round(out3.params['psth'].stderr,6)} \n Seam contribution is fixed thanks to p_s={ratio}p_bulk, p_bulk_th={pbth}")
        plt.legend()
        plt.show()

    if 'correlated contributions full' in fits:
        fit_params3 = Parameters()
        fit_params3.add('alphac', value=1., min=0.0, max=1000.0)
        fit_params3.add('alpha3', value=1., min=0.0, max=100.0)
        fit_params3.add('alpha1', value=1, min=0.0, max=1.0)
        fit_params3.add('alpha2', value=1, min=0.0, max=1.0)
        fit_params3.add('psth', value=1, min=0.0, max=1.0)
        # Values to which we want to compare the fit
        # alpha3test = 0.026
        # alphactest = 372s
        pbth = 7.3e-3
        alpha3test = 0.012
        alphactest = 0.34
        if data_type == 'p fixed':
            p = data_tofit[list(data.keys())[0]][3][0]
        if data_type == 'p_bell fixed':
            p_bell = data_tofit[list(data.keys())[0]][3][0]
        # Definition of the residual function

        def objective_coupled_full(params):
            resid = []
            for d in list(data.keys()):
                p_log = data_tofit[d][1][:-1]
                alpha1 = params['alpha1'].value
                alpha2 = params['alpha2'].value
                pbth = 7.3e-3
                psth = params['psth'].value
                alpha3 = params['alpha3'].value
                alphac = params['alphac'].value
                std_err = data_tofit[d][2][:-1]
                # Objective function is of the form (y-f(x))/std_y
                if data_type == 'p fixed':
                    ps = data_tofit[d][0][:-1]
                    p = data_tofit[d][3][:-1]
                    # for dxd patches:
                if data_type == 'p_bell fixed':
                    p = data_tofit[d][0][:-1]
                    ps = data_tofit[d][3][:-1]
                resid = np.concatenate((resid, list((p_log-coupled_error_model_v2(
                    ps, p, alpha1, alpha2, alpha3, psth,
                    pbth, alphac, d))/std_err)))
            return (resid)
        # Minimize it and report the result
        mini = Minimizer(objective_coupled_full, fit_params3, nan_policy='propagate')
        out3 = mini.minimize(method=method)
        print(fit_report(out3))
        # Calcule de l'intervalle de confiance
        ci = conf_interval(mini, out3)
        report_ci(ci)
        # Plot the result
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, which="major", linestyle="-", linewidth=0.9)
        for d in data_tofit.keys():
            p_log = data_tofit[d][1]
            std_err = data_tofit[d][2]
            alpha3 = out3.params['alpha3'].value
            alphac = out3.params['alphac'].value
            alpha1 = out3.params['alpha1'].value
            alpha2 = out3.params['alpha2'].value
            psth = out3.params['psth'].value
            # p or ps depending of the type of data
            x_fit = np.linspace(
                np.min(data_tofit[d][0]), np.max(data_tofit[d][0]), 100)
            plt.errorbar(data_tofit[d][0], p_log,
                         yerr=std_err, fmt="o", label=f"d={d}", capsize=2, alpha=0.7)
            plt.ylim((1/nb_shots)*1e-2, 1e-1)  # Adjust the limits as needed
            if data_type == 'p fixed':
                plt.plot(x_fit, decoupled_error_model(x_fit, p, 0.29, 0.007, 0.1,
                         0.05, d), "b", alpha=0.5)
                plt.plot(x_fit, coupled_error_model_v2(x_fit, p, alpha1, alpha2,
                         alpha3, psth, pbth, alphac, d), "g")
                plt.legend()
                plt.ylabel('Logical error rate per round')
                plt.xlabel(r'$p_{\text{Bell}}$', fontsize=16)
            if data_type == 'p_bell fixed':
                plt.plot(x_fit, coupled_error_model_v2(p_bell, x_fit, alpha1, alpha2,
                                                       alpha3, psth, pbth, alphac, d), "r")
                # Reference du fit 3d
                plt.plot(x_fit, coupled_error_model_v2(p_bell, x_fit, alpha1,
                         alpha2, alpha3test, psth, pbth, alphactest, d), "g")
                plt.legend()
                plt.xlabel(r'$p_{\text{Bulk}}$', fontsize=16)
                plt.title(r"$p_{\text{Bell}}=$"f"{p_bell}", fontsize=16)
        plt.show()

    print('\n'*2)

# %% Fit 3D


def fit_3D(data_tofit, data, P_bulk, P_seam, fit_interval, fit_3D_type, model=None, plot=True):
    """Perform the different possible 3D fits.

    fit_interval is (pmin,pmax,psmin,psmax, ratio_std_err) defining over which subset of the data
    we fit (relevant region for resource estimation).
    fit_3D_type is a list with the type of fit:
        'naive' : fit the naive ansatz with the two contribution without interaction term.
        'full' : fit the full ansatz (model parameter can be used to change the model).
    If plot is True, also does the final plot the show the fit
    """
    if 'naive' in fit_3D_type:
        # Declare the parameters of the fit
        fit_params3 = Parameters()
        fit_params3.add('alpha2', value=0.5, min=0.0, max=1.0)
        fit_params3.add('alpha1', value=0.5, min=0.0, max=1.0)
        fit_params3.add('pbth', value=1e-2, min=0.0, max=1.0)
        fit_params3.add('psth', value=1e-1, min=0.0, max=1.0)

        def objective_coupled_3d_all_para_naive(params):
            resid = []
            for d in list(data_tofit.keys()):
                if d > 3:
                    # Objective function is of the form (y-f(x))/std_y
                    (pmin, pmax, psmin, psmax, ratio_std_err) = fit_interval
                    # filter the data to only use meaningful points
                    p_log, pb, ps, std_err = filter_data(data_tofit[d], pmin=pmin, pmax=pmax,
                                                         psmin=psmin, psmax=psmax,
                                                         ratio_std_err=ratio_std_err)
                    alpha1 = params['alpha1'].value
                    alpha2 = params['alpha2'].value
                    psth = params['psth'].value
                    pbth = params['pbth'].value
                    resid = np.concatenate((resid, list((p_log-decoupled_error_model(ps,
                                            pb, psth, pbth, alpha1, alpha2, d))/std_err)))
            return (resid)
        # Create the Minimizer
        mini = Minimizer(objective_coupled_3d_all_para_naive, fit_params3, nan_policy='propagate')
        # fit avec leastsq
        out3 = mini.minimize(method='leastsq')
        # out3 = minimize(objective_coupled_3d, fit_params3)
        print(fit_report(out3))
        # Calcule de l'intervalle de confiance
        # ci = conf_interval(mini, out3)
        # report_ci(ci)
        # Reduced chi-squared statistic
        print(r"N-Nparam="+f'{out3.nfree}')

    if 'full' in fit_3D_type:
        # Declare the parameters of the fit
        fit_params3 = Parameters()
        fit_params3.add('alphac', value=0.5, min=0.0, max=1000.0)
        fit_params3.add('alpha3', value=0.5, min=0.0, max=1.0)
        fit_params3.add('alpha2', value=0.5, min=0.0, max=1.0)
        fit_params3.add('alpha1', value=0.5, min=0.0, max=1.0)
        fit_params3.add('pbth', value=1e-2, min=0.0, max=1.0)
        fit_params3.add('psth', value=1e-1, min=0.0, max=1.0)

        def objective_coupled_3d_all_para(params):
            resid = []
            for d in list(data_tofit.keys()):
                if d > 3:
                    # Objective function is of the form (y-f(x))/std_y
                    (pmin, pmax, psmin, psmax, ratio_std_err) = fit_interval
                    p_log, pb, ps, std_err = filter_data(data_tofit[d], pmin=pmin, pmax=pmax,
                                                         psmin=psmin, psmax=psmax,
                                                         ratio_std_err=ratio_std_err)
                    alpha3 = params['alpha3'].value
                    alphac = params['alphac'].value
                    alpha1 = params['alpha1'].value
                    alpha2 = params['alpha2'].value
                    psth = params['psth'].value
                    pbth = params['pbth'].value
                    if model == 'v2':
                        resid = np.concatenate((resid, list((p_log-coupled_error_model_v2(
                            ps, pb, alpha1, alpha2, alpha3,
                            psth, pbth, alphac, d))/std_err)))
                    elif model == 'v1':
                        resid = np.concatenate((resid, list((p_log-coupled_error_model(
                            ps, pb, alpha1, alpha2, alpha3,
                            psth, pbth, alphac, d))/std_err)))
                    else:
                        resid = np.concatenate((resid, list((p_log-coupled_error_model_v2(
                            ps, pb, alpha1, alpha2, alpha3,
                            psth, pbth, alphac, d))/std_err)))
                        print('by default model V2 has been used')
            return (resid)
        # Create the Minimizer
        mini = Minimizer(objective_coupled_3d_all_para, fit_params3, nan_policy='propagate')
        # fit avec leastsq
        out3 = mini.minimize(method='leastsq')
        # out3 = minimize(objective_coupled_3d, fit_params3)
        print(fit_report(out3))
        # Calcule de l'intervalle de confiance
        # ci = conf_interval(mini, out3)
        # report_ci(ci)
        # Reduced chi-squared statistic
        print(r"N-Nparam="+f'{out3.nfree}')

    if plot:
        if 'full' in fit_3D_type:
            threed_full_plot(data_tofit, out3, P_bulk, P_seam, filtered=False, model='v2')
        if 'naive' in fit_3D_type:
            threed_full_plot_naive(data_tofit, out3, P_bulk, P_seam, filtered=False)
    return (data_tofit, out3)


def threed_full_plot(data_tofit, out3, P_bulk, P_seam, filtered=False, model='v2'):
    """Do the plot of the 3d fit where the fitted model display as surfaces and data as points.

    Suited for the full fit.
    Also plot some slice of the 3D space.
    """
    # Paramètres de sortie
    alpha3 = out3.params['alpha3'].value
    alphac = out3.params['alphac'].value
    alpha1 = out3.params['alpha1'].value
    alpha2 = out3.params['alpha2'].value
    psth = out3.params['psth'].value
    pbth = out3.params['pbth'].value
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for d in data_tofit.keys():
        points = []
        p_log = data_tofit[d][2][:]
        std_err = data_tofit[d][3][:]
        ps = data_tofit[d][1][:]
        p = data_tofit[d][0][:]
        for i in range(len(p)):
            if filtered is True:
                if std_err[i] < p_log[i]/2:
                    points.append((p[i], ps[i], p_log[i]))
            else:
                points.append((p[i], ps[i], p_log[i]))
        # Listes contenant des triplets de la forme p_bulk,p_bell,p_logic
        x, y, z = zip(*points)
        # Listes contenant des triplets de la forme p_bulk,p_bell,p_logic
        log_grid_x, log_grid_y = np.mgrid[min(np.log10(x)):max(np.log10(x)):len(P_bulk)*1j,
                                          min(np.log10(y)):max(np.log10(y)):len(P_seam)*1j]
        grid_x, grid_y = 10**log_grid_x, 10**log_grid_y
        # # Liste contenant error_model(p_bulk,p_bell)
        if model == 'v2':
            fitted_z = coupled_error_model_v2(
                grid_y, grid_x, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        elif model == 'v1':
            fitted_z = coupled_error_model(
                grid_y, grid_x, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        else:
            fitted_z = coupled_error_model_v2(
                grid_y, grid_x, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        ax.scatter(np.log10(x), np.log10(y), np.log10(z))
        ax.plot_surface(np.log10(grid_x), np.log10(grid_y), np.log10(
            fitted_z), color=f'C{d}', alpha=0.5, label=f'distance d={d}')

    # Do the log scale by hand
    def log_tick_formatter(val, pos=None):
        # remove int() if you don't use MaxNLocator
        return f"$10^{{{int(val)}}}$"
        # return f"{10**val:.2e}"      # e-Notation

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.yaxis.set_rotate_label(False)  # disable automatic rotation
    ax.xaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r"$P_L^X$", fontsize=20, rotation=0, labelpad=20)
    ax.set_xlabel(r'$p_{\text{bulk}}$', fontsize=20)
    ax.set_ylabel(r'$p_{\text{Bell}}$', fontsize=20, labelpad=10)
    fig.savefig('figures/3d_fit.pdf')
    plt.show()
    plot_sliced_fixed_pbulk(data_tofit, out3, P_bulk, P_seam,
                            fixed_pbulk=1e-3, filtered=filtered, model=model)
    plot_sliced_fixed_pbell(data_tofit, out3, P_bulk, P_seam, fixed_pbell=1e-2,
                            filtered=filtered, model=model)


def plot_sliced_fixed_pbulk(data_tofit, out3, P_bulk, P_seam, fixed_pbulk, filtered, model='v2'):
    """Plot a 2D slice of the 3D fit at a fixed p_bulk value.

    selecting the one closest to `fixed_pbulk`.
    """
    # Extract fitting parameters
    alpha3 = out3.params['alpha3'].value
    alphac = out3.params['alphac'].value
    alpha1 = out3.params['alpha1'].value
    alpha2 = out3.params['alpha2'].value
    psth = out3.params['psth'].value
    pbth = out3.params['pbth'].value

    # Find the closest p_bulk to the fixed value
    closest_pbulk = min(P_bulk, key=lambda x: abs(x - fixed_pbulk))

    plt.figure(figsize=(8, 6))
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow
        "#17becf"   # Cyan
    ]
    markers = [
        "o",  # Circle
        "v",  # Triangle Down
        "^",  # Triangle Up
        "s",  # Square
        "p",  # Pentagon
        "*",  # Star
        "+",  # Plus
        "x",  # X
    ]
    marker_index = 0
    color_index = 0
    for d in data_tofit.keys():
        points = []
        p_log = data_tofit[d][2][:]
        std_err = data_tofit[d][3][:]
        ps = data_tofit[d][1][:]
        p = data_tofit[d][0][:]

        for i in range(len(p)):
            if filtered is True:
                if std_err[i] < p_log[i] / 2:
                    points.append((p[i], ps[i], p_log[i], std_err[i]))
            else:
                points.append((p[i], ps[i], p_log[i], std_err[i]))

        # Convert to numpy array for filtering
        points = np.array(points)
        mask = np.isclose(points[:, 0], closest_pbulk, atol=1e-5)
        filtered_points = points[mask]

        if len(filtered_points) == 0:
            continue  # Skip if no matching points

        y, z, sigma = filtered_points[:, 1], filtered_points[:, 2], filtered_points[:, 3]

        # Sort by increasing p_bell
        sorted_indices = np.argsort(y)
        y_sorted = y[sorted_indices]
        z_sorted = z[sorted_indices]
        sigma_sorted = sigma[sorted_indices]

        # Fit model to extract the expected surface values
        if model == 'v2':
            fitted_z = coupled_error_model_v2(
                y_sorted, closest_pbulk, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        elif model == 'v1':
            fitted_z = coupled_error_model(
                y_sorted, closest_pbulk, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        else:
            fitted_z = coupled_error_model_v2(
                y_sorted, closest_pbulk, alpha1, alpha2, alpha3, psth, pbth, alphac, d)

        color = colors[color_index % len(colors)]
        color_index += 1  # Increment the index
        marker = markers[marker_index % len(markers)]
        marker_index += 1  # Increment the index
        # plt.scatter(y_sorted, z_sorted, label=f'Data (d={d})', marker='o')
        plt.errorbar(y_sorted, z_sorted, yerr=sigma_sorted,
                     fmt=marker, capsize=5, label=f'd={d}', color=color, ecolor=color)
        plt.plot(y_sorted, fitted_z, label=None, color=color, linestyle='--')

    # Add vertical lines at p_Bell = 1e-2 and p_Bell = 5e-2
    plt.axvline(x=1e-2, color='black', linestyle=':')
    plt.axvline(x=5e-2, color='black', linestyle=':')

    # Use log scale
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r'$p_{\text{Bell}}$', fontsize=21)
    plt.ylabel(r'$P_L^X$', fontsize=21, rotation=0, ha='right')
    plt.title(f"Slice at $p_{{\\text{{bulk}}}} = {closest_pbulk:.1e}$", fontsize=18)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.savefig(f'figures/slice_pbulk_{closest_pbulk:.1e}.pdf')
    plt.show()


def plot_sliced_fixed_pbell(data_tofit, out3, P_bulk, P_seam, fixed_pbell, filtered, model='v2'):
    """Plot a 2D slice of the 3D fit at a fixed p_bell value.

    selecting the one closest to `fixed_pbell`.
    """
    # Extract fitting parameters
    alpha3 = out3.params['alpha3'].value
    alphac = out3.params['alphac'].value
    alpha1 = out3.params['alpha1'].value
    alpha2 = out3.params['alpha2'].value
    psth = out3.params['psth'].value
    pbth = out3.params['pbth'].value

    # Find the closest p_bell to the fixed value
    closest_pbell = min(P_seam, key=lambda x: abs(x - fixed_pbell))

    plt.figure(figsize=(8, 6))

    for d in data_tofit.keys():
        points = []
        p_log = data_tofit[d][2][:]
        std_err = data_tofit[d][3][:]
        ps = data_tofit[d][1][:]
        p = data_tofit[d][0][:]

        for i in range(len(p)):
            if filtered is True:
                if std_err[i] < p_log[i] / 2:
                    points.append((p[i], ps[i], p_log[i]))
            else:
                points.append((p[i], ps[i], p_log[i]))

        # Convert to numpy array for filtering
        points = np.array(points)
        mask = np.isclose(points[:, 1], closest_pbell, atol=1e-6)
        filtered_points = points[mask]

        if len(filtered_points) == 0:
            continue  # Skip if no matching points

        x, y, z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

        # Sort by increasing p_bulk for smooth plotting
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        z_sorted = z[sorted_indices]

        # Fit model to extract the expected surface values
        if model == 'v2':
            fitted_z = coupled_error_model_v2(
                closest_pbell, x_sorted, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        elif model == 'v1':
            fitted_z = coupled_error_model(
                closest_pbell, x_sorted, alpha1, alpha2, alpha3, psth, pbth, alphac, d)
        else:
            fitted_z = coupled_error_model_v2(
                closest_pbell, x_sorted, alpha1, alpha2, alpha3, psth, pbth, alphac, d)

        plt.scatter(x_sorted, z_sorted, label=f'Data (d={d})', marker='o')
        plt.plot(x_sorted, fitted_z, label=f'Fit (d={d})', linestyle='--')

    # Use log scale
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r'$p_{\text{bulk}}$', fontsize=16)
    plt.ylabel(r'$p_{\text{L}}$', fontsize=16)
    plt.title(f"Slice at $p_{{\\text{{bell}}}} = {closest_pbell:.1e}$", fontsize=18)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.savefig(f'figures/slice_pbell_{closest_pbell:.1e}.pdf')
    plt.show()


def threed_full_plot_naive(data_tofit, out3, P_bulk, P_seam, filtered=False):
    """Do the plot of the 3d fit where the fitted model display as surfaces and data as points.

    Suited for the naive fit.
    Also plot some slice of the 3D space.
    """
    # Paramètres de sortie
    alpha1 = out3.params['alpha1'].value
    alpha2 = out3.params['alpha2'].value
    psth = out3.params['psth'].value
    pbth = out3.params['pbth'].value
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for d in data_tofit.keys():
        points = []
        p_log = data_tofit[d][2][:]
        std_err = data_tofit[d][3][:]
        ps = data_tofit[d][1][:]
        p = data_tofit[d][0][:]
        for i in range(len(p)):
            if filtered is True:
                if std_err[i] < p_log[i]/2:
                    points.append((p[i], ps[i], p_log[i]))
            else:
                points.append((p[i], ps[i], p_log[i]))
        # Listes contenant des triplets de la forme p_bulk,p_bell,p_logic
        x, y, z = zip(*points)
        # Listes contenant des triplets de la forme p_bulk,p_bell,p_logic
        log_grid_x, log_grid_y = np.mgrid[min(np.log10(x)):max(np.log10(x)):len(P_bulk)*1j,
                                          min(np.log10(y)):max(np.log10(y)):len(P_seam)*1j]
        grid_x, grid_y = 10**log_grid_x, 10**log_grid_y
        # # Liste contenant error_model(p_bulk,p_bell)
        fitted_z = decoupled_error_model(grid_y, grid_x, psth, pbth, alpha1, alpha2, d)
        ax.scatter(np.log10(x), np.log10(y), np.log10(z))
        ax.plot_surface(np.log10(grid_x), np.log10(grid_y), np.log10(
            fitted_z), color=f'C{d}', alpha=0.5, label=f'distance d={d}')

    # Do the log scale by hand
    def log_tick_formatter(val, pos=None):
        # remove int() if you don't use MaxNLocator
        return f"$10^{{{int(val)}}}$"
        # return f"{10**val:.2e}"      # e-Notation

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.yaxis.set_rotate_label(False)  # disable automatic rotation
    ax.xaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r"$P_L^X$", fontsize=20, rotation=0, labelpad=20)
    ax.set_xlabel(r'$p_{\text{bulk}}$', fontsize=20)
    ax.set_ylabel(r'$p_{\text{Bell}}$', fontsize=20, labelpad=10)
    fig.savefig('figures/3d_fit_naive.pdf')
    plt.show()
    plot_sliced_fixed_pbulk_naive(data_tofit, out3, P_bulk, P_seam,
                                  fixed_pbulk=1e-3, filtered=filtered)


def plot_sliced_fixed_pbulk_naive(data_tofit, out3, P_bulk, P_seam, fixed_pbulk=1e-3, filtered=False):
    """Plot a 2D slice of the 3D fit at a fixed p_bulk value.

    selecting the one closest to `fixed_pbulk`.
    """
    # Extract fitting parameters
    alpha1 = out3.params['alpha1'].value
    alpha2 = out3.params['alpha2'].value
    psth = out3.params['psth'].value
    pbth = out3.params['pbth'].value

    # Find the closest p_bulk to the fixed value
    closest_pbulk = min(P_bulk, key=lambda x: abs(x - fixed_pbulk))

    plt.figure(figsize=(8, 6))
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow
        "#17becf"   # Cyan
    ]
    markers = [
        "o",  # Circle
        "v",  # Triangle Down
        "^",  # Triangle Up
        "s",  # Square
        "p",  # Pentagon
        "*",  # Star
        "+",  # Plus
        "x",  # X
    ]
    marker_index = 0
    color_index = 0
    for d in data_tofit.keys():
        points = []
        p_log = data_tofit[d][2][:]
        std_err = data_tofit[d][3][:]
        ps = data_tofit[d][1][:]
        p = data_tofit[d][0][:]

        for i in range(len(p)):
            if filtered is True:
                if std_err[i] < p_log[i] / 2:
                    points.append((p[i], ps[i], p_log[i], std_err[i]))
            else:
                points.append((p[i], ps[i], p_log[i], std_err[i]))

        # Convert to numpy array for filtering
        points = np.array(points)
        mask = np.isclose(points[:, 0], closest_pbulk, atol=1e-5)
        filtered_points = points[mask]

        if len(filtered_points) == 0:
            continue  # Skip if no matching points

        y, z, sigma = filtered_points[:, 1], filtered_points[:, 2], filtered_points[:, 3]

        # Sort by increasing p_bell for smooth plotting
        sorted_indices = np.argsort(y)
        y_sorted = y[sorted_indices]
        z_sorted = z[sorted_indices]
        sigma_sorted = sigma[sorted_indices]

        # Fit model to extract the expected surface values
        fitted_z = decoupled_error_model(
            y_sorted, closest_pbulk, psth, pbth, alpha1, alpha2, d)

        color = colors[color_index % len(colors)]
        color_index += 1  # Increment the index
        marker = markers[marker_index % len(markers)]
        marker_index += 1  # Increment the index
        # plt.scatter(y_sorted, z_sorted, label=f'Data (d={d})', marker='o')
        plt.errorbar(y_sorted, z_sorted, yerr=sigma_sorted,
                     fmt=marker, capsize=5, label=f'd={d}', color=color, ecolor=color)
        plt.plot(y_sorted, fitted_z, label=None, color=color, linestyle='--')

    # Add vertical lines at p_Bell = 1e-2 and p_Bell = 5e-2
    plt.axvline(x=1e-2, color='black', linestyle=':')
    plt.axvline(x=5e-2, color='black', linestyle=':')

    # Use log scale
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r'$p_{\text{Bell}}$', fontsize=21)
    plt.ylabel(r'$P_L^X$', fontsize=21, rotation=0, ha='right')
    plt.title(f"Slice at $p_{{\\text{{bulk}}}} = {closest_pbulk:.1e}$ with naive fit", fontsize=18)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.savefig(f'figures/slice_pbulk_{closest_pbulk:.1e}_naive_fit.pdf')
    plt.show()
