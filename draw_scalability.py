""" Perform simulation to estimate cost of VoD service across three major cloud CDN providers"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data_file = "scalability.csv"
    df = pd.read_csv(data_file)

    # Visualization

    sns.set(style="whitegrid")
    # fig, ax = plt.subplots()
    # ax.set(xscale="log", yscale="log")
    # sns.set_context("poster")
    sns.set_context("poster", font_scale=1, rc={"lines.linewidth": 2.5})
    g = sns.factorplot(x="User demand", y="Session crashes", hue="Provider", data=df, col="Region", kind="bar",
                       palette="muted", ci=95,
                       n_boot=1000, size=6, aspect=1, sharex=True)

    g.despine(left=True)
    g.set_ylabels("Number of Session Crashes")
    g.set_xlabels("User demand")
    g.set_xticklabels([r'2 to 256', r'512', r'1024'])
    g._legend.set_title('Provider')
    plt.show()
    g.savefig("scalability_session_crashes.pdf")


    sns.set(style="whitegrid")
    # fig, ax = plt.subplots()
    # ax.set(xscale="log", yscale="log")
    # sns.set_context("poster")
    sns.set_context("poster", font_scale=1, rc={"lines.linewidth": 2.5})
    g = sns.factorplot(x="User demand", y="QoE", hue="Provider", data=df, col="Region", kind="bar",
                       palette="muted", ci=95,
                       n_boot=1000, size=6, aspect=1, sharex=True)

    g.despine(left=True)
    g.set_ylabels("Number of Session Crashes")
    g.set_xlabels("User demand")
    g.set_xticklabels([r'2 to 256', r'512', r'1024'])
    g._legend.set_title('Provider')
    plt.show()
    g.savefig("scalability_QoE.pdf")
