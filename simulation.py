""" Perform simulation to estimate cost of VoD service across three major cloud CDN providers"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

egress_data_range_tbl = {
    "azure_premium" : [10, 50, 150, 500, 1024, 5120],
    "azure": [10, 50, 150, 500, 1024, 5120],
    "aws": [10, 50, 150, 500, 1024, 5120],
    "google": [10, 150, 1000]
}

egress_gb_cost_table = {
    "azure_premium": {
        "North America": [0.17, 0.15, 0.13, 0.11, 0.10, 0.09, 0.09],
        "Europe": [0.17, 0.15, 0.13, 0.11, 0.10, 0.09, 0.09],
        "Asia Pacific": [0.25, 0.22, 0.19, 0.16, 0.14, 0.12, 0.12]
    },
    "azure": {
        "North America": [0.087, 0.08, 0.06, 0.04, 0.03, 0.025, 0.025],
        "Europe": [0.087, 0.08, 0.06, 0.04, 0.03, 0.025, 0.025],
        "Asia Pacific": [0.138, 0.13, 0.12, 0.10, 0.08, 0.07, 0.07]
    },
    "aws": {
        "North America": [0.085, 0.080, 0.060, 0.040, 0.030, 0.025, 0.020],
        "Europe": [0.085, 0.080, 0.060, 0.040, 0.030, 0.025, 0.020 ],
        "Asia Pacific": [0.140, 0.135, 0.120, 0.100, 0.080, 0.070, 0.060]
    },
    "google": {
        "North America": [0.08, 0.055, 0.03, 0.02],
        "Europe": [0.08, 0.055, 0.03, 0.02],
        "Asia Pacific": [0.09, 0.06, 0.05, 0.04]
    }
}

cachefill_data_range_tbl = {
    "azure_premium" : [5, 10, 50, 150, 500],
    "azure": [5, 10, 50, 150, 500],
    "aws": [],
    "google": []
}

cachefill_gb_cost_tbl = {
    "azure_premium": {
        "North America":[0.0, 0.087, 0.083, 0.07, 0.05, 0.05],
        "Europe":[0.0, 0.087, 0.083, 0.07, 0.05, 0.05],
        "Asia Pacific":[0, 0.138, 0.135, 0.13, 0.12, 0.12]
    },
    "azure": {
        "North America":[0.0, 0.087, 0.083, 0.07, 0.05, 0.05],
        "Europe":[0.0, 0.087, 0.083, 0.07, 0.05, 0.05],
        "Asia Pacific":[0, 0.138, 0.135, 0.13, 0.12, 0.12]
    },
    "aws": 0,
    "google":{
        "North America": 0.04,
        "Europe": 0.05,
        "Asia Pacific": 0.06
    }
}

httprequests_cost_tbl = {
    "azure_premium":0,
    "azure":0,
    "aws":{
        "North America":0.0075,
        "Europe":0.009,
        "Asia Pacific":0.009
    },
    "google":0.0075
}


## Compute the cost of traffic in progressive pricing schemes.
def compute_incremental_cost(data_gb, data_tb_range, egress_gb_cost):
    data_tb = data_gb / 1024

    effective_data_gb_range = [max_bound*1024 for max_bound in data_tb_range if max_bound <= data_tb]
    range_sz = len(effective_data_gb_range)
    effective_data_gb_cost = egress_gb_cost[:range_sz+1]
    # print("Data range size: %d; Data cost size: %d", range_sz, len(effective_data_gb_cost))

    cost = 0
    pre_bound = 0
    for i, gb_bound in enumerate(effective_data_gb_range):
        cost += (gb_bound - pre_bound) * effective_data_gb_cost[i]
        pre_bound = gb_bound

    cost += (data_gb - pre_bound) * effective_data_gb_cost[range_sz]
    return cost

## Compute the egress data cost in one CDN region for a particular CDN provider
def get_regional_egress_data_cost(data_gb, provider, region):
    global egress_data_range_tbl, egress_gb_cost_table
    data_tb_range = egress_data_range_tbl[provider]
    egress_gb_cost = egress_gb_cost_table[provider][region]

    regional_data_cost = compute_incremental_cost(data_gb, data_tb_range, egress_gb_cost)
    return regional_data_cost

## Compute the cache fill data cost in one CDN region for a particular CDN provider
def get_regional_cachefill_data_cost(cachefill_data_gb, provider, region):
    global cachefill_data_range_tbl, cachefill_gb_cost_tbl
    provider_cost = cachefill_gb_cost_tbl[provider]
    if isinstance(provider_cost, dict):
        provider_regional_cost = provider_cost[region]

        if isinstance(provider_regional_cost, list):
            cur_data_range = cachefill_data_range_tbl[provider]
            cur_cachefill_gb_cost = cachefill_gb_cost_tbl[provider][region]
            cur_cost = compute_incremental_cost(cachefill_data_gb, cur_data_range, cur_cachefill_gb_cost)
            return cur_cost
        else:
            return provider_regional_cost*cachefill_data_gb

    else:
        return cachefill_data_gb * provider_cost

def get_regional_httprequests_cost(request_num, provider, region):
    global httprequests_cost_tbl
    provider_cost = httprequests_cost_tbl[provider]
    if isinstance(provider_cost,dict):
        per_tenthousand_cost = provider_cost[region]
        return per_tenthousand_cost * request_num / 10000
    else:
        return provider_cost * request_num / 10000

def consumption_region(region, data, hours):
    total_data = data * hours
    total_data_df = pd.DataFrame(data={'total_data': total_data, 'hours': hours}, index=region)
    total_data_region = total_data_df.groupby(level=0).sum()

    #print(total_data_region.columns.tolist())
    #print(total_data_region.index.tolist())

    return total_data_region


def get_total_cost(provider, total_data_region, hit_rate=0):
    total_cost = 0
    for region in total_data_region.index.tolist():
        regional_egress_data = total_data_region.loc[region].total_data
        regional_cachefill_data = regional_egress_data * hit_rate
        regional_total_requests = total_data_region.loc[region].hours * 3600 / 5.0
        regional_cost = get_regional_egress_data_cost(regional_egress_data, provider, region) \
                        + get_regional_cachefill_data_cost(regional_cachefill_data, provider, region) \
                        + get_regional_httprequests_cost(regional_total_requests, provider, region)
        total_cost += regional_cost
    return total_cost

def generate_sample(n_samples, bit_rate):
    # prob = (np.random.multinomial(100, [1/6.]*6, size=1)/100).flatten().tolist()

    # Define the users belong to multinomial distributed regions
    region_names = ["North America", "Europe", "Asia Pacific"]

    ## Multinomial distribution
    # regions = np.random.choice(3, n_samples,
    #                           p=(np.random.multinomial(len(region_names), [1 / 3.] * 3, size=1) / n_samples).flatten().tolist()) + 1

    ## Uniform distribution
    regions = np.random.choice(region_names, n_samples)

    # Define the QoE proxied by the bit rate (bits per second)
    bit_rate_index = np.random.geometric(p=0.7, size=n_samples) - 1
    bit_rate_index[np.where(bit_rate_index > 9)] = 9

    qoe = bit_rate[bit_rate_index - 1].values

    streaming_data = qoe / 8 / 1024 / 1024 / 1024 * 3600 # GB/hour

    # Define the watching hours as the poisson distributed with the average of 28 hours per month
    # according to a Netflix study https://techcrunch.com/2017/04/10/netflix-reaches-75-of-u-s-streaming-service-viewers-but-youtube-is-catching-up/
    watching_hours = np.random.negative_binomial(n=3.2, p=0.1, size=n_samples)
    return regions, streaming_data, watching_hours



def estimate_cost(provider, total_data_region, hit_rate=0):
    cost = cost_region1 = cost_region2 = cost_region3 = 0

    total_data_region1 = float(total_data_region.loc[1].total_data)
    total_data_region2 = float(total_data_region.loc[2].total_data)
    total_data_region3 = float(total_data_region.loc[3].total_data)

    hours_region1 = float(total_data_region.loc[1].hours)
    hours_region2 = float(total_data_region.loc[2].hours)
    hours_region3 = float(total_data_region.loc[3].hours)

    data_region1_tb = total_data_region1 / 1024
    data_region2_tb = total_data_region2 / 1024
    data_region3_tb = total_data_region3 / 1024

    if provider == 'azure_premium':
        if data_region1_tb <= 10:
            cost_region1 = 0.17 * total_data_region1
        elif 10 < data_region1_tb <= 50:
            cost_region1 = 0.17 * 10 * 1024 + 0.15 * total_data_region1 - 0.15 * 10 * 1024
        elif 50 < data_region1_tb <= 150:
            cost_region1 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * total_data_region1 - 0.13 * 50 * 1024
        elif 150 < data_region1_tb <= 500:
            cost_region1 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * 100 * 1024 \
                           + 0.11 * total_data_region1 - 0.11 * 150 * 1024
        elif 500 < data_region1_tb <= 1024:
            cost_region1 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * 100 * 1024 \
                           + 0.11 * 350 * 1024 + 0.10 * total_data_region1 - 0.10 * 350 * 1024
        elif 1024 < data_region1_tb <= 5120:
            cost_region1 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * 100 * 1024 \
                           + 0.11 * 350 * 1024 + 0.10 * 524 * 1024 + 0.09 * total_data_region1 \
                           - 0.09 * 524 * 1024

        if data_region2_tb <= 10:
            cost_region2 = 0.17 * total_data_region2
        elif 10 < data_region2_tb <= 50:
            cost_region2 = 0.17 * 10 * 1024 + 0.15 * total_data_region2 - 0.15 * 10 * 1024
        elif 50 < data_region2_tb <= 150:
            cost_region2 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * total_data_region2 - 0.13 * 50 * 1024
        elif 150 < data_region2_tb <= 500:
            cost_region2 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * 100 * 1024 \
                           + 0.11 * total_data_region2 - 0.11 * 150 * 1024
        elif 500 < data_region2_tb <= 1024:
            cost_region2 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * 100 * 1024 \
                           + 0.11 * 350 * 1024 + 0.10 * total_data_region2 - 0.10 * 350 * 1024
        elif 1024 < data_region2_tb <= 5120:
            cost_region2 = 0.17 * 10 * 1024 + 0.15 * 40 * 1024 + 0.13 * 100 * 1024 \
                           + 0.11 * 350 * 1024 + 0.10 * 524 * 1024 + 0.09 * total_data_region2 \
                           - 0.09 * 524 * 1024

        if data_region3_tb <= 10:
            cost_region3 = 0.25 * total_data_region3
        elif 10 < data_region3_tb <= 50:
            cost_region3 = 0.25 * 10 * 1024 + 0.15 * total_data_region3 - 0.22 * 10 * 1024
        elif 50 < data_region3_tb <= 150:
            cost_region3 = 0.25 * 10 * 1024 + 0.22 * 40 * 1024 + 0.19 * total_data_region3 - 0.19 * 50 * 1024
        elif 150 < data_region3_tb <= 500:
            cost_region3 = 0.25 * 10 * 1024 + 0.22 * 40 * 1024 + 0.19 * 100 * 1024 \
                           + 0.16 * total_data_region3 - 0.16 * 150 * 1024
        elif 500 < data_region3_tb <= 1024:
            cost_region3 = 0.25 * 10 * 1024 + 0.22 * 40 * 1024 + 0.19 * 100 * 1024 \
                           + 0.16 * 350 * 1024 + 0.14 * total_data_region3 - 0.14 * 350 * 1024
        elif 1024 < data_region3_tb <= 5120:
            cost_region3 = 0.25 * 10 * 1024 + 0.22 * 40 * 1024 + 0.19 * 100 * 1024 \
                           + 0.16 * 350 * 1024 + 0.14 * 524 * 1024 + 0.12 * total_data_region3 \
                           - 0.12 * 524 * 1024

    if provider == 'azure':
        if data_region1_tb <= 10:
            cost_region1 = 0.087 * total_data_region1
        elif 10 < data_region1_tb <= 50:
            cost_region1 = 0.087 * 10 * 1024 + 0.08 * total_data_region1 - 0.08 * 10 * 1024
        elif 50 < data_region1_tb <= 150:
            cost_region1 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * total_data_region1 - 0.06 * 50 * 1024
        elif 150 < data_region1_tb <= 500:
            cost_region1 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 \
                           + 0.04 * total_data_region1 - 0.04 * 150 * 1024
        elif 500 < data_region1_tb <= 1024:
            cost_region1 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 \
                           + 0.04 * 350 * 1024 + 0.03 * total_data_region1 - 0.03 * 350 * 1024
        elif 1024 < data_region1_tb:
            cost_region1 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 \
                           + 0.04 * 350 * 1024 + 0.03 * 524 * 1024 + 0.025 * total_data_region1 \
                           - 0.025 * 524 * 1024

        if data_region2_tb <= 10:
            cost_region2 = 0.087 * total_data_region2
        elif 10 < data_region2_tb <= 50:
            cost_region2 = 0.087 * 10 * 1024 + 0.08 * total_data_region2 - 0.08 * 10 * 1024
        elif 50 < data_region2_tb <= 150:
            cost_region2 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * total_data_region2 - 0.06 * 50 * 1024
        elif 150 < data_region2_tb <= 500:
            cost_region2 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 \
                           + 0.04 * total_data_region2 - 0.04 * 150 * 1024
        elif 500 < data_region2_tb <= 1024:
            cost_region2 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 \
                           + 0.04 * 350 * 1024 + 0.03 * total_data_region2 - 0.03 * 350 * 1024
        elif 1024 < data_region2_tb :
            cost_region2 = 0.087 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 \
                           + 0.04 * 350 * 1024 + 0.03 * 524 * 1024 + 0.025 * total_data_region2 \
                           - 0.025 * 524 * 1024

        if data_region3_tb <= 10:
            cost_region3 = 0.138 * total_data_region3
        elif 10 < data_region3_tb <= 50:
            cost_region3 = 0.138 * 10 * 1024 + 0.13 * total_data_region3 - 0.13 * 10 * 1024
        elif 50 < data_region3_tb <= 150:
            cost_region3 = 0.138 * 10 * 1024 + 0.13 * 40 * 1024 + 0.12 * total_data_region3 - 0.12 * 50 * 1024
        elif 150 < data_region3_tb <= 500:
            cost_region3 = 0.138 * 10 * 1024 + 0.13 * 40 * 1024 + 0.12 * 100 * 1024 \
                           + 0.1 * total_data_region3 - 0.1 * 150 * 1024
        elif 500 < data_region3_tb <= 1024:
            cost_region3 = 0.138 * 10 * 1024 + 0.13 * 40 * 1024 + 0.12 * 100 * 1024 \
                           + 0.1 * 350 * 1024 + 0.08 * total_data_region3 - 0.08 * 350 * 1024
        elif 1024 < data_region3_tb :
            cost_region3 = 0.138 * 10 * 1024 + 0.13 * 40 * 1024 + 0.12 * 100 * 1024 \
                           + 0.1 * 350 * 1024 + 0.08 * 524 * 1024 + 0.07 * total_data_region3 \
                           - 0.07 * 524 * 1024



    if provider == 'aws':
        if data_region1_tb <= 10:
            cost_region1 = 0.085 * total_data_region1 + 0.0075 * hours_region1 * 3600 / 50000
        elif 10 < data_region1_tb <= 50:
            cost_region1 = 0.085 * 10 * 1024 + 0.08 * total_data_region1 - 0.08 * 10 * 1024 \
                           + 0.0075 * hours_region1 * 3600 / 50000
        elif 50 < data_region1_tb <= 150:
            cost_region1 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * total_data_region1 \
                           - 0.06 * 50 * 1024 + 0.0075 * hours_region1 * 3600 / 50000
        elif 150 < data_region1_tb <= 500:
            cost_region1 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 + 0.04 * total_data_region1 \
                           - 0.04 * 150 * 1024 + 0.0075 * hours_region1 * 3600 / 50000
        elif 500 < data_region1_tb <= 1024:
            cost_region1 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 + 0.04 * 350 * 1024 \
                           + 0.03 * total_data_region1 - 0.03 * 500 * 1024 + 0.0075 * hours_region1 * 3600 / 50000
        elif 1024 < data_region1_tb :
            cost_region1 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 + 0.04 * 350 * 1024 \
                           + 0.03 * 524 * 1024 + 0.025 * total_data_region1 - 0.025 * 1024 * 1024 \
                           + 0.0075 * hours_region1 * 3600 / 50000

        if data_region2_tb <= 10:
            cost_region2 = 0.085 * total_data_region2 + 0.009 * hours_region2 * 3600 / 50000
        elif 10 < data_region2_tb <= 50:
            cost_region2 = 0.085 * 10 * 1024 + 0.08 * total_data_region2 - 0.08 * 10 * 1024 \
                           + 0.009 * hours_region2 * 3600 / 50000
        elif 50 < data_region2_tb <= 150:
            cost_region2 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * total_data_region2 \
                           - 0.06 * 50 * 1024 + 0.009 * hours_region2 * 3600 / 50000
        elif 150 < data_region2_tb <= 500:
            cost_region2 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 + 0.04 * total_data_region2 \
                           - 0.04 * 150 * 1024 + 0.009 * hours_region2 * 3600 / 50000
        elif 500 < data_region2_tb <= 1024:
            cost_region2 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 + 0.04 * 350 * 1024 \
                           + 0.03 * total_data_region2 - 0.03 * 500 * 1024 + 0.009 * hours_region2 * 3600 / 50000
        elif 1024 < data_region2_tb:
            cost_region2 = 0.085 * 10 * 1024 + 0.08 * 40 * 1024 + 0.06 * 100 * 1024 + 0.04 * 350 * 1024 \
                           + 0.03 * 524 * 1024 + 0.025 * total_data_region2 - 0.025 * 1024 * 1024 \
                           + 0.009 * hours_region2 * 3600 / 50000

        if data_region3_tb <= 10:
            cost_region3 = 0.14 * total_data_region3 + 0.009 * hours_region3 * 3600 / 50000
        elif 10 < data_region3_tb <= 50:
            cost_region3 = 0.14 * 10 * 1024 + 0.135 * total_data_region3 - 0.135 * 10 * 1024 \
                           + 0.009 * hours_region3 * 3600 / 50000
        elif 50 < data_region3_tb <= 150:
            cost_region3 = 0.14 * 10 * 1024 + 0.135 * 40 * 1024 + 0.12 * total_data_region3 \
                           - 0.12 * 50 * 1024 + 0.009 * hours_region3 * 3600 / 50000
        elif 150 < data_region3_tb <= 500:
            cost_region3 = 0.14 * 10 * 1024 + 0.135 * 40 * 1024 + 0.12 * 100 * 1024 + 0.1 * total_data_region3 \
                           - 0.1 * 150 * 1024 + 0.009 * hours_region3 * 3600 / 50000
        elif 500 < data_region3_tb <= 1024:
            cost_region3 = 0.14 * 10 * 1024 + 0.135 * 40 * 1024 + 0.12 * 100 * 1024 + 0.1 * 350 * 1024 \
                           + 0.08 * total_data_region2 - 0.08 * 500 * 1024 + 0.009 * hours_region3 * 3600 / 50000
        elif 1024 < data_region3_tb :
            cost_region3 = 0.14 * 10 * 1024 + 0.135 * 40 * 1024 + 0.12 * 100 * 1024 + 0.1 * 350 * 1024 \
                           + 0.08 * 524 * 1024 + 0.07 * total_data_region3 - 0.07 * 1024 * 1024 \
                           + 0.009 * hours_region3 * 3600 / 50000

    if provider == 'google':

        # hit_rate = 0.5

        if data_region1_tb <= 10:
            cost_region1 = 0.08 * total_data_region1 + (1 - hit_rate) * 0.04 * total_data_region1
        elif 10 < data_region1_tb <= 150:
            cost_region1 = 0.08 * 10 * 1024 + 0.055 * total_data_region1 - 0.055 * 10 * 1024 \
                           + (1 - hit_rate) * 0.04 * total_data_region1
        elif 150 < data_region1_tb <= 1000:
            cost_region1 = 0.08 * 10 * 1024 + 0.055 * 140 * 1024 + 0.03 * total_data_region1 \
                           - 0.03 * 150 * 1024 + (1 - hit_rate) * 0.04 * total_data_region1
        elif data_region1_tb > 1000:
            cost_region1 = 0.08 * 10 * 1024 + 0.055 * 140 * 1024 + 0.03 * 850 * 1024 \
                           + 0.02 * total_data_region1 - 0.02 * 1000 * 1024 + \
                           (1 - hit_rate) * 0.04 * total_data_region1

        if data_region2_tb <= 10:
            cost_region2 = 0.08 * total_data_region2 + (1 - hit_rate) * 0.05 * total_data_region2
        elif 10 < data_region2_tb <= 150:
            cost_region2 = 0.08 * 10 * 1024 + 0.055 * total_data_region2 - 0.055 * 10 * 1024 \
                           + (1 - hit_rate) * 0.05 * total_data_region2
        elif 150 < data_region2_tb <= 1000:
            cost_region2 = 0.08 * 10 * 1024 + 0.055 * 140 * 1024 + 0.03 * total_data_region2 \
                           - 0.03 * 150 * 1024 + (1 - hit_rate) * 0.05 * total_data_region2
        elif data_region2_tb > 1000:
            cost_region2 = 0.08 * 10 * 1024 + 0.055 * 140 * 1024 + 0.03 * 850 * 1024 \
                           + 0.02 * total_data_region2 - 0.02 * 1000 * 1024 + \
                           (1 - hit_rate) * 0.05 * total_data_region2

        if data_region3_tb <= 10:
            cost_region3 = 0.09 * total_data_region3 + (1 - hit_rate) * 0.05 * total_data_region3
        elif 10 < data_region3_tb <= 150:
            cost_region3 = 0.09 * 10 * 1024 + 0.06 * total_data_region3 - 0.06 * 10 * 1024 \
                           + (1 - hit_rate) * 0.06 * total_data_region3
        elif 150 < data_region3_tb <= 1000:
            cost_region3 = 0.09 * 10 * 1024 + 0.06 * 140 * 1024 + 0.05 * total_data_region3 \
                           - 0.05 * 150 * 1024 + (1 - hit_rate) * 0.06 * total_data_region3
        elif data_region3_tb > 1000:
            cost_region3 = 0.09 * 10 * 1024 + 0.06 * 140 * 1024 + 0.05 * 850 * 1024 \
                           + 0.04 * total_data_region3 - 0.04 * 1000 * 1024 + \
                           (1 - hit_rate) * 0.06 * total_data_region3

    cost = cost_region1 + cost_region2 + cost_region3
    return cost

def experiment_over_scales():
    n_samples = [1000, 10000, 100000, 1000000, 10000000]  # size of sample
    n_experiments = 3  # number of experiments
    hit_rate_range = [0, 0.5, 0.9, 1]  # Cache of hit rate
    total_cost = []
    bit_rate = pd.Series({'9': 313769.0, '8': 722907.0, '7': 1292014.0, '6': 1917529.0, '5': 2925460.0,
                          '4': 4023268.0, '3': 5477193.0, '2': 8769548.0, '1': 11365148.0})

    for p in ['aws', 'azure', 'google']:
        for sample in n_samples:
            for i in range(n_experiments):
                for hit_rate in hit_rate_range:
                    print("Provider: %s; User demand: %d; Experiment %d; Hit rate: %.1f" % (p, sample, i, hit_rate))
                    regions, streaming_data, watching_hours = generate_sample(n_samples=sample, bit_rate=bit_rate)
                    total_data_region = consumption_region(region=regions, data=streaming_data, hours=watching_hours)
                    total_cost.append([p, sample, hit_rate,
                                       get_total_cost(provider=p, total_data_region=total_data_region,
                                                     hit_rate=hit_rate)])

    total_cost_df = pd.DataFrame(total_cost, columns=['provider', 'samples', 'hit_rate', 'cost'])

    total_cost_df.loc[total_cost_df['provider'] == 'aws', 'provider'] = 'Amazon CloudFront'
    total_cost_df.loc[total_cost_df['provider'] == 'azure', 'provider'] = 'Azure CDN'
    total_cost_df.loc[total_cost_df['provider'] == 'google', 'provider'] = 'Google Cloud CDN'

    total_cost_df.to_csv("total_cost.csv")

if __name__ == "__main__":

    data_file = "total_cost.csv"
    total_cost_df = pd.read_csv(data_file)

    # Visualization

    sns.set(style="whitegrid")
    # fig, ax = plt.subplots()
    # ax.set(xscale="log", yscale="log")
    # sns.set_context("poster")
    sns.set_context("poster", font_scale=1, rc={"lines.linewidth": 2.5})
    g = sns.factorplot(x="samples", y="cost", hue="provider", data=total_cost_df.loc[total_cost_df["hit_rate"] != 0.9], col="hit_rate", kind="point",
                       palette="muted", ci=95,
                       n_boot=1000, size=6, aspect=1, sharex=True)

    g.despine(left=True)
    g.set_ylabels("Estimated cost ($)")
    g.set_xlabels("User demand")
    g.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$'])
    g.set(yscale="log")
    g._legend.set_title('Provider')
    plt.show()
    g.savefig("cost.pdf")
